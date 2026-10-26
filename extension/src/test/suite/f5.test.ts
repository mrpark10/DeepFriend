import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';
import { api, waitFor, wsPath } from './helpers';

const FIX = path.join(__dirname, '..', '..', '..', 'src', 'test', 'fixtures', 'runs');

function install(fixture: string, id: string, startedAt: string, extra: Record<string, unknown> = {}, code?: Record<string, string>): string {
  const dir = wsPath('.deepfriend', 'runs', id);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const meta = JSON.parse(fs.readFileSync(path.join(FIX, fixture, 'meta.json'), 'utf8'));
  Object.assign(meta, { id, startedAt, status: 'finished', starred: false }, extra);
  fs.writeFileSync(path.join(dir, 'meta.json'), JSON.stringify(meta));
  fs.copyFileSync(path.join(FIX, fixture, 'metrics.jsonl'), path.join(dir, 'metrics.jsonl'));
  for (const [rel, text] of Object.entries(code ?? {})) {
    fs.mkdirSync(path.dirname(path.join(dir, 'code', rel)), { recursive: true });
    fs.writeFileSync(path.join(dir, 'code', rel), text);
  }
  return dir;
}

suite('F5 experiments', () => {
  let dirA: string;
  let dirB: string;

  suiteSetup(async () => {
    const a = await api();
    dirA = install('healthy', '20200101-000001-a5a5', '2020-01-01T00:00:01Z', { name: 'exp-A', config: { lr: 0.001, batch_size: 32 } }, { 'f5/train.py': 'lr = 0.001\n' });
    dirB = install('overfit', '20200101-000002-b5b5', '2020-01-01T00:00:02Z', { name: 'exp-B', config: { lr: 0.01, batch_size: 32 } }, { 'f5/train.py': 'lr = 0.01\n' });
    await waitFor(() => a.doctor.watcher.store.get(dirA) && a.doctor.watcher.store.get(dirB));
  });

  test('the tree lists runs newest first with their headline metric', async () => {
    const a = await api();
    const labels = a.runs.tree.getChildren().map((i) => i.label);
    assert.ok(labels.indexOf('exp-B') >= 0 && labels.indexOf('exp-B') < labels.indexOf('exp-A'), labels.join(','));
    const itemA = a.runs.tree.getChildren().find((i) => i.label === 'exp-A')!;
    assert.ok(String(itemA.description).includes('val_acc'), String(itemA.description));
  });

  test('star persists to meta.json and changes the icon', async () => {
    const a = await api();
    const item = a.runs.tree.getChildren().find((i) => i.label === 'exp-A')!;
    await vscode.commands.executeCommand('deepfriend.toggleStar', item);
    assert.strictEqual(JSON.parse(fs.readFileSync(path.join(dirA, 'meta.json'), 'utf8')).starred, true);
    const again = a.runs.tree.getChildren().find((i) => i.label === 'exp-A')!;
    assert.strictEqual((again.iconPath as vscode.ThemeIcon).id, 'star-full');
  });

  test('restore copies the snapshot back and backs up the current file', async () => {
    const a = await api();
    fs.mkdirSync(wsPath('f5'), { recursive: true });
    fs.writeFileSync(wsPath('f5', 'train.py'), 'lr = 999\n');
    const item = a.runs.tree.getChildren().find((i) => i.label === 'exp-B')!;
    a.runs.testHooks.autoConfirm = true;
    try {
      await vscode.commands.executeCommand('deepfriend.restoreRun', item);
    } finally {
      a.runs.testHooks.autoConfirm = false;
    }
    assert.strictEqual(fs.readFileSync(wsPath('f5', 'train.py'), 'utf8'), 'lr = 0.01\n');
    const backups = fs.readdirSync(wsPath('.deepfriend', 'backups'));
    const latest = backups.sort().pop()!;
    assert.strictEqual(fs.readFileSync(wsPath('.deepfriend', 'backups', latest, 'f5', 'train.py'), 'utf8'), 'lr = 999\n');
  });

  test('compare opens a panel with the settings difference', async () => {
    const a = await api();
    const ra = a.doctor.watcher.store.get(dirA)!;
    const rb = a.doctor.watcher.store.get(dirB)!;
    const panel = a.runs.compare(ra, rb);
    assert.ok(panel.webview.html.includes('<tr class="diff"><td>lr</td><td>0.001</td><td>0.01</td></tr>'), panel.webview.html.slice(0, 2000));
    assert.ok(panel.webview.html.includes('data-file="0"'));
    panel.dispose();
  });

  test('delete removes the run folder and the tree entry', async () => {
    const a = await api();
    const tmp = install('chance', '20200101-000003-c5c5', '2020-01-01T00:00:03Z', { name: 'exp-delete' });
    await waitFor(() => a.runs.tree.getChildren().find((i) => i.label === 'exp-delete'));
    const item = a.runs.tree.getChildren().find((i) => i.label === 'exp-delete')!;
    a.runs.testHooks.autoConfirm = true;
    try {
      await vscode.commands.executeCommand('deepfriend.deleteRun', item);
    } finally {
      a.runs.testHooks.autoConfirm = false;
    }
    assert.strictEqual(fs.existsSync(tmp), false);
    assert.ok(!a.runs.tree.getChildren().some((i) => i.label === 'exp-delete'));
  });
});
