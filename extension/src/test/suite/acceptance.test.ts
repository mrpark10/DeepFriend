import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import { buildIssueUrl } from '../../reportUrl';
import { api, shellCommand, testShell, waitFor, wsPath } from './helpers';

const EXAMPLES = path.join(__dirname, '..', '..', '..', '..', 'examples');

suite('acceptance: a student’s first training run', () => {
  test('zero-install watch() in the VS Code terminal → doctor diagnoses NaN → run is listed', async function () {
    this.timeout(120_000);
    const a = await api();
    const dir = wsPath('student');
    fs.mkdirSync(dir, { recursive: true });
    fs.copyFileSync(path.join(EXAMPLES, '05_doctor', '_data.py'), path.join(dir, '_data.py'));
    fs.copyFileSync(path.join(EXAMPLES, '05_doctor', 'nan.py'), path.join(dir, 'nan.py'));
    // The terminal gets PYTHONPATH from the extension: no `pip install deepfriend` anywhere.
    const term = vscode.window.createTerminal({ name: 'student', shellPath: testShell(), cwd: dir });
    try {
      term.show();
      const si = await waitFor(() => term.shellIntegration, 20000);
      si.executeCommand(shellCommand(process.env.DEEPFRIEND_TEST_PYTHON!, 'nan.py'));
      const msg = await waitFor(() => {
        const m = a.doctor.view.lastMessage;
        return m?.run?.title === 'nan' && m.findings.length ? m : undefined;
      }, 90_000, 200);
      assert.strictEqual(msg.findings[0].id, 'nan');
      assert.ok(msg.findings[0].what.includes('NaN'));
      await waitFor(() => a.runs.tree.getChildren().find((i) => i.label === 'nan'), 30_000);
    } finally {
      term.dispose();
    }
  });

  test('the prefilled issue link survives VS Code URI handling without double encoding', () => {
    const url = buildIssueUrl({ title: '[오류] 테스트', intro: '무엇을?', extVersion: '0.1.0', vscodeVersion: '1.0', os: 'x', python: null, torch: null, tensorflow: null, error: 'a%b c' });
    const viaVscode = vscode.Uri.parse(url).toString(true);
    assert.strictEqual(new URL(viaVscode).searchParams.get('title'), '[오류] 테스트');
    assert.ok(new URL(viaVscode).searchParams.get('body')!.includes('a%b c'));
  });
});
