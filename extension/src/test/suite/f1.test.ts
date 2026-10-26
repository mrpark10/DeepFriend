import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import { api, waitFor, wsPath } from './helpers';

const HEADER = 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n';

const STEM = HEADER + `# df: input (32, 3, 224, 224)
class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

    def forward(self, x):
        x = self.conv(x)
        return x
`;

const BUG = HEADER + `# df: input (64, 1, 28, 28)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(10000, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)
`;

const NEEDS_INPUT = HEADER + `from torchvision import datasets
train = datasets.MNIST("data")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.flatten(1))
`;

function lineOf(text: string, needle: string): number {
  return text.split('\n').findIndex((l) => l.includes(needle));
}

async function openFresh(name: string, content: string): Promise<vscode.TextEditor> {
  const file = wsPath('f1', name);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, content);
  const doc = await vscode.workspace.openTextDocument(file);
  return vscode.window.showTextDocument(doc);
}

async function hints(doc: vscode.TextDocument): Promise<vscode.InlayHint[]> {
  const r = new vscode.Range(0, 0, doc.lineCount, 0);
  return (await vscode.commands.executeCommand<vscode.InlayHint[]>('vscode.executeInlayHintProvider', doc.uri, r)) ?? [];
}

function label(h: vscode.InlayHint): string {
  return typeof h.label === 'string' ? h.label : h.label.map((p) => p.value).join('');
}

function deepfriendDiags(doc: vscode.TextDocument): vscode.Diagnostic[] {
  return vscode.languages.getDiagnostics(doc.uri).filter((d) => d.source === 'DeepFriend');
}

suite('F1 shape hints', () => {
  suiteSetup(async () => {
    const a = await api();
    await a.ensureServer();
    await vscode.workspace.getConfiguration('deepfriend').update('language', 'ko', vscode.ConfigurationTarget.Global);
  });

  test('shows the spec example shape at the end of the conv line', async () => {
    const ed = await openFresh('stem.py', STEM);
    const hs = await waitFor(async () => {
      const h = await hints(ed.document);
      return h.length ? h : null;
    });
    const convLine = lineOf(STEM, 'self.conv = ');
    const h = hs.find((x) => x.position.line === convLine);
    assert.ok(h, 'no hint on conv line');
    assert.strictEqual(label(h!), '→ (32, 64, 112, 112)');
    assert.strictEqual(h!.position.character, ed.document.lineAt(convLine).text.length);
  });

  test('Linear mismatch: Korean message, quick fix, and it disappears after fixing', async () => {
    const ed = await openFresh('bug.py', BUG);
    const doc = ed.document;
    const diags = await waitFor(() => {
      const d = deepfriendDiags(doc);
      return d.length ? d : null;
    });
    const fcLine = lineOf(BUG, 'self.fc1 =');
    const def = diags.find((d) => d.range.start.line === fcLine)!;
    assert.strictEqual(def.message, '앞 레이어 출력은 12544개인데 Linear는 10000개를 기대해요. in_features를 12544로 바꿀까요?');
    assert.strictEqual(doc.getText(def.range), '10000');
    const actions = await vscode.commands.executeCommand<vscode.CodeAction[]>('vscode.executeCodeActionProvider', doc.uri, def.range, vscode.CodeActionKind.QuickFix.value);
    const fix = actions.find((a) => a.title === 'in_features를 12544로 바꾸기');
    assert.ok(fix && fix.edit, 'quick fix missing');
    await vscode.workspace.applyEdit(fix!.edit!);
    await doc.save();
    await waitFor(() => deepfriendDiags(doc).length === 0);
    assert.ok(doc.getText().includes('nn.Linear(12544, 10)'));
  });

  test('CodeLens asks for the input shape and inserting it shows hints', async () => {
    const ed = await openFresh('needs_input.py', NEEDS_INPUT);
    const doc = ed.document;
    const lenses = await waitFor(async () => {
      const l = await vscode.commands.executeCommand<vscode.CodeLens[]>('vscode.executeCodeLensProvider', doc.uri);
      return l && l.length ? l : null;
    });
    const lens = lenses.find((l) => l.command?.command === 'deepfriend.setInputShape')!;
    assert.ok(lens.command!.title.includes('입력 크기 정하기'));
    assert.strictEqual(lens.command!.arguments![2], '(B, 1, 28, 28)');
    const args = [...lens.command!.arguments!, '(8, 1, 28, 28)'];
    await vscode.commands.executeCommand('deepfriend.setInputShape', ...args);
    assert.ok(doc.getText().includes('# df: input (8, 1, 28, 28)\nclass MLP'));
    await doc.save();
    const hs = await waitFor(async () => {
      const h = await hints(doc);
      return h.length ? h : null;
    });
    assert.ok(hs.some((h) => label(h) === '→ (8, 10)'));
  });

  test('switching to English re-renders messages', async () => {
    const bug2 = BUG.replace('10000', '9999');
    const ed = await openFresh('bug_en.py', bug2);
    await waitFor(() => deepfriendDiags(ed.document).length > 0);
    await vscode.workspace.getConfiguration('deepfriend').update('language', 'en', vscode.ConfigurationTarget.Global);
    try {
      await waitFor(() => deepfriendDiags(ed.document).some((d) => d.message.startsWith('The previous layer outputs 12544 values')));
    } finally {
      await vscode.workspace.getConfiguration('deepfriend').update('language', 'ko', vscode.ConfigurationTarget.Global);
    }
  });

  test('files without deep-learning imports are never analyzed', async () => {
    const ed = await openFresh('plain.py', 'print("hello")\n');
    await new Promise((r) => setTimeout(r, 800));
    const a = await api();
    assert.strictEqual(a.store.get(ed.document.uri.toString()), undefined);
  });

  test('hints update within 1 second after saving', async () => {
    const ed = await openFresh('timing.py', STEM);
    const doc = ed.document;
    await waitFor(async () => (await hints(doc)).length > 0);
    const convLine = lineOf(STEM, 'self.conv = ');
    await ed.edit((b) => b.replace(doc.lineAt(convLine).range, '        self.conv = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)'));
    const start = Date.now();
    await doc.save();
    await waitFor(async () => (await hints(doc)).some((h) => label(h) === '→ (32, 16, 112, 112)'), 5000, 20);
    const elapsed = Date.now() - start;
    assert.ok(elapsed < 1000, `took ${elapsed}ms`);
  });
});
