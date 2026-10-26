import * as assert from 'assert';
import * as fs from 'fs';
import * as vscode from 'vscode';
import { CONCEPTS } from '../../f7/concepts';
import { api, wsPath } from './helpers';

const SRC = `import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            3,
            16,
            kernel_size=3,
        )
`;

async function hoverText(doc: vscode.TextDocument, pos: vscode.Position): Promise<string> {
  const hovers = await vscode.commands.executeCommand<vscode.Hover[]>('vscode.executeHoverProvider', doc.uri, pos);
  return hovers.flatMap((h) => h.contents.map((c) => (typeof c === 'string' ? c : (c as vscode.MarkdownString).value))).join('\n');
}

suite('F7 concept hover', () => {
  let doc: vscode.TextDocument;

  suiteSetup(async () => {
    await api();
    fs.mkdirSync(wsPath('f7'), { recursive: true });
    fs.writeFileSync(wsPath('f7', 'hover.py'), SRC);
    doc = await vscode.workspace.openTextDocument(wsPath('f7', 'hover.py'));
    await vscode.window.showTextDocument(doc);
  });

  test('hovering nn.Conv2d shows the beginner Korean explanation', async () => {
    const line = SRC.split('\n').findIndex((l) => l.includes('nn.Conv2d('));
    const col = SRC.split('\n')[line].indexOf('Conv2d') + 2;
    const text = await hoverText(doc, new vscode.Position(line, col));
    assert.ok(text.includes(CONCEPTS['torch.Conv2d'].what.beginner.ko), text);
  });

  test('hovering kernel_size in a multi-line call explains that parameter first', async () => {
    const line = SRC.split('\n').findIndex((l) => l.includes('kernel_size='));
    const text = await hoverText(doc, new vscode.Position(line, 14));
    assert.ok(text.includes('**`kernel_size`** — ' + CONCEPTS['torch.Conv2d'].params.kernel_size.ko), text);
  });

  test('the level setting switches to the normal explanation', async () => {
    await vscode.workspace.getConfiguration('deepfriend').update('level', 'normal', vscode.ConfigurationTarget.Global);
    try {
      const line = SRC.split('\n').findIndex((l) => l.includes('nn.Conv2d('));
      const col = SRC.split('\n')[line].indexOf('Conv2d') + 2;
      const text = await hoverText(doc, new vscode.Position(line, col));
      assert.ok(text.includes(CONCEPTS['torch.Conv2d'].what.normal.ko), text);
    } finally {
      await vscode.workspace.getConfiguration('deepfriend').update('level', undefined, vscode.ConfigurationTarget.Global);
    }
  });
});
