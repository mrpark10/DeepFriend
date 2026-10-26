import * as assert from 'assert';
import * as vscode from 'vscode';
import { api, waitFor, wsPath } from './helpers';

suite('analysis server', () => {
  test('starts with the configured python and reports framework versions', async () => {
    const doc = await vscode.workspace.openTextDocument(wsPath('model.py'));
    await vscode.window.showTextDocument(doc);
    const a = await api();
    assert.strictEqual(await a.ensureServer(), true);
    await waitFor(() => a.status() === 'ready');
    const info = a.client.info!;
    assert.strictEqual(info.version, '0.1.0');
    assert.ok(info.torch, 'torch version should be reported');
    assert.strictEqual(a.python()!.source, 'setting');
  });

  test('restart command brings the server back', async () => {
    const a = await api();
    await vscode.commands.executeCommand('deepfriend.restartServer');
    await waitFor(() => a.status() === 'ready');
    const r = await a.client.request<{ text: string }>('echo', { text: '박준호' });
    assert.deepStrictEqual(r, { text: '박준호' });
  });
});
