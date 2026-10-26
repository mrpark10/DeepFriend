import * as assert from 'assert';
import * as path from 'path';
import * as vscode from 'vscode';

suite('activation', () => {
  test('activates on a python file', async () => {
    const ws = vscode.workspace.workspaceFolders![0].uri.fsPath;
    const doc = await vscode.workspace.openTextDocument(path.join(ws, 'model.py'));
    await vscode.window.showTextDocument(doc);
    const ext = vscode.extensions.getExtension('mrpark10.deepfriend');
    assert.ok(ext, 'extension not found');
    await ext!.activate();
    assert.strictEqual(ext!.isActive, true);
  });
});
