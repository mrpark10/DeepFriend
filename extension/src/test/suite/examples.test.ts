import * as assert from 'assert';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as vscode from 'vscode';
import { api } from './helpers';

suite('examples & onboarding', () => {
  test('openExample copies every example into a fresh folder', async () => {
    await api();
    const base = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'df-ex-')), 'DeepFriend-examples');
    const dest1 = await vscode.commands.executeCommand<string>('deepfriend.openExample', '01_mnist_mlp.py', base, false);
    const dest2 = await vscode.commands.executeCommand<string>('deepfriend.openExample', '01_mnist_mlp.py', base, false);
    assert.strictEqual(dest1, base);
    assert.strictEqual(dest2, base + '-2');
    for (const f of ['01_mnist_mlp.py', '02_cifar_cnn.py', '03_keras_cnn.py', '04_shape_error.py', 'README.md', path.join('05_doctor', 'nan.py'), path.join('05_doctor', '_data.py')]) {
      assert.ok(fs.existsSync(path.join(dest1, f)), f);
    }
  });

  test('the walkthrough is contributed with four steps', () => {
    const ext = vscode.extensions.getExtension('mrpark10.deepfriend')!;
    const wt = ext.packageJSON.contributes.walkthroughs[0];
    assert.strictEqual(wt.id, 'deepfriend.start');
    assert.strictEqual(wt.steps.length, 4);
    for (const s of wt.steps) {
      assert.ok(fs.existsSync(path.join(ext.extensionPath, s.media.markdown)), s.media.markdown);
    }
  });
});
