import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import { explain } from '../../f3/errorDictionary';
import type { ReportedError } from '../../f3/errorHub';
import { findTracebacks } from '../../f3/traceback';
import { api, waitFor, wsPath } from './helpers';

const TB = `Traceback (most recent call last):
  File "${wsPath('f3', 'train.py').replace(/\\/g, '\\\\')}", line 3, in <module>
    out = fc(x)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x12544 and 10000x10)
`;

suite('F3 error explanations', () => {
  suiteSetup(async () => {
    await api();
    fs.mkdirSync(wsPath('f3'), { recursive: true });
    fs.writeFileSync(wsPath('f3', 'train.py'), 'import torch\nfc = torch.nn.Linear(10000, 10)\nout = fc(torch.zeros(32, 12544))\n');
  });

  test('the card shows the Korean explanation with the numbers filled in', async () => {
    const a = await api();
    const err = findTracebacks(TB)[0];
    a.errors.card.show({ error: err, source: 'terminal' }, explain(err));
    const html = a.errors.card.lastHtml;
    assert.ok(html.includes('크기 불일치: Linear 층에 들어가는 숫자 개수가 달라요'), html);
    assert.ok(html.includes('앞 레이어가 12544개의 숫자를 내보내는데'), html);
    assert.ok(html.includes('무슨 뜻이에요') && html.includes('왜 그래요') && html.includes('이렇게 해보세요'));
    assert.ok(html.includes('data-action="goto"'));
    assert.ok(html.includes("Content-Security-Policy"));
  });

  test('go to error line opens the file at that line', async () => {
    const err = findTracebacks(TB)[0];
    await vscode.commands.executeCommand('deepfriend.goToErrorLine', err.frames[0]);
    const ed = await waitFor(() => (vscode.window.activeTextEditor?.document.fileName.endsWith('train.py') ? vscode.window.activeTextEditor : undefined));
    assert.strictEqual(ed.selection.start.line, 2);
  });

  test('an error.json written by deepfriend.watch() reaches the hub', async () => {
    const a = await api();
    const seen: ReportedError[] = [];
    const sub = a.errors.hub.onError((e) => seen.push(e));
    try {
      const dir = wsPath('.deepfriend', 'runs', '20261026-120000-abcd');
      fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(path.join(dir, 'error.json'), JSON.stringify({ type: 'ValueError', message: 'from run file ' + Date.now(), frames: [], traceback: '' }));
      await waitFor(() => seen.find((s) => s.source === 'runFile'));
    } finally {
      sub.dispose();
    }
  });

  test('notebook cell error outputs reach the hub', async () => {
    const a = await api();
    const seen: ReportedError[] = [];
    const sub = a.errors.hub.onError((e) => seen.push(e));
    try {
      await vscode.commands.executeCommand(
        'dfTest.errorNotebook',
        'RuntimeError',
        'mat1 and mat2 shapes cannot be multiplied (1x10 and 20x5)',
        'RuntimeError                              Traceback (most recent call last)\nCell In[3], line 5\nRuntimeError: mat1 and mat2 shapes cannot be multiplied (1x10 and 20x5)',
      );
      const got = await waitFor(() => seen.find((s) => s.source === 'notebook'));
      assert.strictEqual(got.error.type, 'RuntimeError');
      assert.strictEqual(explain(got.error)?.id, 'shape_matmul');
      assert.deepStrictEqual(got.error.frames[0], { file: 'Cell In[3]', line: 5, name: '' });
    } finally {
      sub.dispose();
    }
  });

  test('install refuses suspicious package names', async () => {
    const before = vscode.tasks.taskExecutions.length;
    await vscode.commands.executeCommand('deepfriend.installModule', 'evil; rm -rf /');
    assert.strictEqual(vscode.tasks.taskExecutions.length, before);
  });
});
