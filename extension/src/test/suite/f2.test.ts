import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import { api, waitFor, wsPath } from './helpers';

const FIX = path.join(__dirname, '..', '..', '..', 'src', 'test', 'fixtures', 'runs');

function installRun(fixture: string, id: string, patch: Record<string, unknown> = {}): string {
  const dir = wsPath('.deepfriend', 'runs', id);
  fs.mkdirSync(dir, { recursive: true });
  const meta = JSON.parse(fs.readFileSync(path.join(FIX, fixture, 'meta.json'), 'utf8'));
  Object.assign(meta, { id, status: 'finished', startedAt: new Date().toISOString() }, patch);
  fs.copyFileSync(path.join(FIX, fixture, 'metrics.jsonl'), path.join(dir, 'metrics.jsonl'));
  if (fs.existsSync(path.join(FIX, fixture, 'samples.json'))) {
    fs.copyFileSync(path.join(FIX, fixture, 'samples.json'), path.join(dir, 'samples.json'));
  }
  fs.writeFileSync(path.join(dir, 'meta.json'), JSON.stringify(meta));
  return dir;
}

suite('F2 training doctor', () => {
  test('rewriting only meta.json of an existing run refreshes the panel (regression)', async () => {
    const a = await api();
    const dir = installRun('healthy', '20261026-195900-ffff', { name: 'first-name' });
    await waitFor(() => (a.doctor.view.lastMessage?.run?.title === 'first-name' ? true : undefined));
    installRun('healthy', '20261026-195900-ffff', { name: 'renamed' });
    await waitFor(() => (a.doctor.view.lastMessage?.run?.title === 'renamed' ? true : undefined));
    assert.ok(dir);
  });

  test('a NaN run shows the NaN card in Korean', async () => {
    const a = await api();
    installRun('nan', '20261026-200000-aaaa', { name: 'nan-run' });
    const msg = await waitFor(() => (a.doctor.view.lastMessage?.run?.title === 'nan-run' ? a.doctor.view.lastMessage : undefined));
    assert.strictEqual(msg.findings[0].id, 'nan');
    assert.ok(msg.findings[0].what.startsWith('loss가 NaN(숫자 아님)이 됐어요'));
    assert.deepStrictEqual(msg.findings[0].actions, ['학습률 1/10로 바꾸기', 'eps 더하는 코드 보기']);
    assert.ok(msg.chart.x.length > 0);
  });

  test('a healthy run says everything is fine', async () => {
    const a = await api();
    installRun('healthy', '20261026-200100-bbbb', { name: 'healthy-run' });
    const msg = await waitFor(() => (a.doctor.view.lastMessage?.run?.title === 'healthy-run' ? a.doctor.view.lastMessage : undefined));
    assert.deepStrictEqual(msg.findings, []);
    assert.strictEqual(msg.note, '지금까지는 문제가 보이지 않아요. 잘 배우고 있어요!');
  });

  test('"divide learning rate by 10" edits the training script', async () => {
    const a = await api();
    await a.ensureServer();
    fs.mkdirSync(wsPath('f2'), { recursive: true });
    fs.writeFileSync(wsPath('f2', 'train_nan.py'), 'import torch\nmodel = torch.nn.Linear(2, 2)\nopt = torch.optim.SGD(model.parameters(), lr=500.0)\n');
    const dir = installRun('nan', '20261026-200200-cccc', { name: 'fix-run', script: 'f2/train_nan.py', config: { lr: 500.0, optimizer: 'SGD' } });
    const run = await waitFor(() => a.doctor.watcher.store.get(dir));
    a.doctor.testHooks.autoConfirm = true;
    try {
      const ok = await a.doctor.applyAction({ id: 'nan', args: {}, actions: [] }, { kind: 'setLr', factor: 0.1 }, run);
      assert.strictEqual(ok, true);
    } finally {
      a.doctor.testHooks.autoConfirm = false;
    }
    assert.ok(fs.readFileSync(wsPath('f2', 'train_nan.py'), 'utf8').includes('lr=50.0'));
  });

  test('snippets open as a new Python document', async () => {
    const a = await api();
    const dir = installRun('overfit', '20261026-200300-dddd');
    const run = await waitFor(() => a.doctor.watcher.store.get(dir));
    await a.doctor.applyAction({ id: 'overfit', args: {}, actions: [] }, { kind: 'snippet', id: 'earlyStopping' }, run);
    const doc = vscode.window.visibleTextEditors.map((e) => e.document).find((d) => d.isUntitled && d.getText().includes('best_val'));
    assert.ok(doc, 'snippet document not opened');
  });

  test('openDoctor pins a run', async () => {
    const a = await api();
    const dir = installRun('chance', '20261026-190000-eeee', { name: 'old-chance' });
    await waitFor(() => a.doctor.watcher.store.get(dir));
    await vscode.commands.executeCommand('deepfriend.openDoctor', dir);
    assert.strictEqual(a.doctor.view.lastMessage?.run?.title, 'old-chance');
    assert.strictEqual(a.doctor.view.lastMessage?.findings[0].id, 'chance');
    a.doctor.watcher.pinned = undefined;
  });
});
