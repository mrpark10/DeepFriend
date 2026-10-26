import * as assert from 'assert';
import * as fs from 'fs';
import * as vscode from 'vscode';
import { api, waitFor, wsPath } from './helpers';

const SRC = `import torch
import torch.nn as nn
import torch.nn.functional as F


class Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


# df: input (1, 1, 32, 32)
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
`;

const lineOf = (needle: string) => SRC.split('\n').findIndex((l) => l.includes(needle));

suite('F4 model structure view', () => {
  let editor: vscode.TextEditor;

  suiteSetup(async () => {
    const a = await api();
    await a.ensureServer();
    fs.mkdirSync(wsPath('f4'), { recursive: true });
    fs.writeFileSync(wsPath('f4', 'lenet.py'), SRC);
    editor = await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(wsPath('f4', 'lenet.py')));
    await waitFor(() => a.store.get(editor.document.uri.toString())?.models.find((m) => m.name === 'LeNet' && m.blocks.length));
  });

  test('cursor inside LeNet shows its 7 blocks and exact parameter count', async () => {
    const a = await api();
    const l = lineOf('self.conv2 =');
    editor.selection = new vscode.Selection(l, 0, l, 0);
    const msg = await waitFor(() => (a.modelView.lastMessage?.model?.name === 'LeNet' ? a.modelView.lastMessage : undefined));
    assert.strictEqual(msg.model!.blocks.length, 7);
    assert.strictEqual(msg.totalParams, 61706);
    assert.strictEqual(msg.model!.paramsText, '파라미터 61.7K개');
    assert.deepStrictEqual(msg.model!.blocks.map((b) => b.name), ['conv1', 'pool', 'conv2', 'pool', 'fc1', 'fc2', 'fc3']);
    assert.strictEqual(msg.model!.blocks[0].outShape, '(1, 6, 28, 28)');
    assert.strictEqual(msg.activeLine, l);
  });

  test('moving the cursor into another class switches the model', async () => {
    const a = await api();
    const l = lineOf('self.fc = nn.Linear(4, 2)');
    editor.selection = new vscode.Selection(l, 0, l, 0);
    await waitFor(() => (a.modelView.lastMessage?.model?.name === 'Small' ? true : undefined));
  });

  test('clicking a block moves the editor to that line', async () => {
    const a = await api();
    const target = lineOf('self.fc3 =');
    editor.selection = new vscode.Selection(0, 0, 0, 0);
    await a.modelView.handleMessage({ type: 'reveal', line: target });
    await waitFor(() => vscode.window.activeTextEditor?.selection.active.line === target);
  });
});
