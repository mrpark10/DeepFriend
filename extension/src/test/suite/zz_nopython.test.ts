import * as assert from 'assert';
import * as fs from 'fs';
import * as vscode from 'vscode';
import { CONCEPTS } from '../../f7/concepts';
import { explain } from '../../f3/errorDictionary';
import { findTracebacks } from '../../f3/traceback';
import { api, waitFor, wsPath } from './helpers';

// Runs only in the "no Python installed" system test (DEEPFRIEND_EXPECT_NO_PYTHON=1).
(process.env.DEEPFRIEND_EXPECT_NO_PYTHON ? suite : suite.skip)('system: no Python on this computer', () => {
  test('status says Python is needed, nothing crashes', async () => {
    const a = await api();
    fs.mkdirSync(wsPath('f_nopy'), { recursive: true });
    fs.writeFileSync(wsPath('f_nopy', 'm.py'), 'import torch\nimport torch.nn as nn\nlayer = nn.Conv2d(3, 8, 3)\n');
    const doc = await vscode.workspace.openTextDocument(wsPath('f_nopy', 'm.py'));
    await vscode.window.showTextDocument(doc);
    assert.strictEqual(await a.ensureServer(), false);
    await waitFor(() => a.status() === 'noPython');
    assert.strictEqual(a.python(), null);
  });

  test('concept hovers still work offline without Python', async () => {
    const doc = await vscode.workspace.openTextDocument(wsPath('f_nopy', 'm.py'));
    const hovers = await vscode.commands.executeCommand<vscode.Hover[]>('vscode.executeHoverProvider', doc.uri, new vscode.Position(2, 13));
    const text = hovers.flatMap((h) => h.contents.map((c) => (typeof c === 'string' ? c : (c as vscode.MarkdownString).value))).join('\n');
    assert.ok(text.includes(CONCEPTS['torch.Conv2d'].what.beginner.ko));
  });

  test('error explanations still work without Python', async () => {
    const a = await api();
    const err = findTracebacks('Traceback (most recent call last):\n  File "/x/train.py", line 3, in <module>\nModuleNotFoundError: No module named \'torch\'\n')[0];
    a.errors.card.show({ error: err, source: 'terminal' }, explain(err));
    assert.ok(a.errors.card.lastHtml.includes('torch'));
  });
});
