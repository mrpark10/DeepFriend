import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { buildChart } from '../f2/chart';
import { projectRoot, RunWatcher } from '../f2';
import type { RunData, RunMeta } from '../f2/runStore';
import { getLanguage, t } from '../i18n/i18n';
import { escapeHtml, webviewHtml } from '../webview/html';
import { codeFiles, configDiff, headlineMetric } from './compareData';

export const f5TestHooks = { autoConfirm: false };

class RunItem extends vscode.TreeItem {
  constructor(public readonly run: RunData) {
    super(RunItem.label(run.meta), vscode.TreeItemCollapsibleState.None);
    const m = run.meta;
    const when = new Date(m.startedAt);
    const time = isNaN(when.getTime()) ? '' : `${when.getMonth() + 1}/${when.getDate()} ${String(when.getHours()).padStart(2, '0')}:${String(when.getMinutes()).padStart(2, '0')}`;
    const metric = headlineMetric(m);
    this.description = [time, metric ? `${metric.key} ${metric.text}` : ''].filter(Boolean).join(' · ');
    this.tooltip = new vscode.MarkdownString(
      [`**${RunItem.label(m)}**`, t(`doctor.status.${m.status}` as never), m.script ? `\`${m.script}\`` : '', ...Object.entries(m.config ?? {}).map(([k, v]) => `- ${k}: ${String(v)}`)].filter(Boolean).join('\n\n'),
    );
    const icon = m.starred ? 'star-full' : m.status === 'running' ? 'sync~spin' : m.status === 'failed' ? 'error' : m.status === 'stopped' ? 'debug-stop' : 'pass';
    this.iconPath = new vscode.ThemeIcon(icon);
    this.contextValue = m.starred ? 'deepfriendRun.starred' : 'deepfriendRun';
    this.id = run.dir;
    this.command = { command: 'deepfriend.openRun', title: '', arguments: [this] };
  }

  static label(m: RunMeta): string {
    return m.name || (m.script ? path.basename(m.script) : m.id);
  }
}

export class RunsTreeProvider implements vscode.TreeDataProvider<RunItem> {
  private changed = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this.changed.event;

  constructor(private watcher: RunWatcher) {}

  refresh(): void {
    this.changed.fire();
  }

  runs(): RunData[] {
    return this.watcher.store
      .all()
      .filter((r) => r.meta.record !== false && path.basename(path.dirname(r.dir)) === 'runs' && fs.existsSync(r.dir))
      .sort((a, b) => (a.meta.startedAt < b.meta.startedAt ? 1 : -1));
  }

  getTreeItem(e: RunItem): vscode.TreeItem {
    return e;
  }

  getChildren(): RunItem[] {
    return this.runs().map((r) => new RunItem(r));
  }
}

function writeMeta(dir: string, patch: Partial<RunMeta>): void {
  const file = path.join(dir, 'meta.json');
  const meta = JSON.parse(fs.readFileSync(file, 'utf8')) as RunMeta;
  Object.assign(meta, patch);
  const tmp = file + '.ext.tmp';
  fs.writeFileSync(tmp, JSON.stringify(meta, null, 2));
  fs.renameSync(tmp, file);
}

async function confirm(message: string, yes: string): Promise<boolean> {
  if (f5TestHooks.autoConfirm) {
    return true;
  }
  return (await vscode.window.showWarningMessage(message, { modal: true }, yes)) === yes;
}

/** Copy a run's code snapshot back into the project, after backing up the current files. */
export async function restoreRun(run: RunData): Promise<string[]> {
  const root = projectRoot(run.dir);
  const files = codeFiles(run.dir, run.dir);
  if (!files.length) {
    void vscode.window.showWarningMessage(t('runs.restore.none'));
    return [];
  }
  if (!(await confirm(t('runs.restore.confirm', { n: files.length, files: files.slice(0, 5).join(', ') + (files.length > 5 ? ' …' : '') }), t('runs.restore.yes')))) {
    return [];
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const backup = path.join(root, '.deepfriend', 'backups', stamp);
  for (const rel of files) {
    const target = path.join(root, ...rel.split('/'));
    if (fs.existsSync(target)) {
      fs.mkdirSync(path.dirname(path.join(backup, rel)), { recursive: true });
      fs.copyFileSync(target, path.join(backup, ...rel.split('/')));
    }
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.copyFileSync(path.join(run.dir, 'code', ...rel.split('/')), target);
  }
  void vscode.window.showInformationMessage(t('runs.restore.done', { n: files.length, backup: path.relative(root, backup) }));
  return files;
}

function comparePanel(ctx: vscode.ExtensionContext, a: RunData, b: RunData): vscode.WebviewPanel {
  const panel = vscode.window.createWebviewPanel('deepfriend.compare', t('runs.compare.title'), vscode.ViewColumn.Active, {
    enableScripts: true,
    localResourceRoots: [vscode.Uri.joinPath(ctx.extensionUri, 'media')],
  });
  const nameA = RunItem.label(a.meta);
  const nameB = RunItem.label(b.meta);
  const rows = configDiff(a.meta, b.meta)
    .map((r) => `<tr class="${r.same ? 'same' : 'diff'}"><td>${escapeHtml(r.key)}</td><td>${escapeHtml(r.a)}</td><td>${escapeHtml(r.b)}</td></tr>`)
    .join('');
  const files = codeFiles(a.dir, b.dir);
  const fileButtons = files.map((f, i) => `<button class="secondary" data-file="${i}">${escapeHtml(f)}</button>`).join('');
  const data = {
    a: { name: nameA, chart: buildChart(a.series), epochs: a.series.epochs },
    b: { name: nameB, chart: buildChart(b.series), epochs: b.series.epochs },
    L: { trainLoss: t('runs.compare.trainLoss'), valLoss: t('runs.compare.valLoss'), step: t('doctor.step'), epoch: t('runs.compare.epoch') },
  };
  const body = `<h1>${escapeHtml(t('runs.compare.heading', { a: nameA, b: nameB }))}</h1>
<h2>${escapeHtml(t('runs.compare.settings'))}</h2>
<table class="diff-table"><thead><tr><th></th><th>A · ${escapeHtml(nameA)}</th><th>B · ${escapeHtml(nameB)}</th></tr></thead><tbody>${rows}</tbody></table>
<h2>${escapeHtml(t('runs.compare.charts'))}</h2>
<div id="charts"></div>
<h2>${escapeHtml(t('runs.compare.code'))}</h2>
<div class="actions">${fileButtons || `<p class="muted">${escapeHtml(t('runs.compare.noCode'))}</p>`}</div>
<script type="application/json" id="compare-data">${JSON.stringify(data).replace(/</g, '\\u003c')}</script>`;
  panel.webview.html = webviewHtml(panel.webview, ctx.extensionUri, {
    title: t('runs.compare.title'),
    body,
    scripts: ['media/vendor/uPlot.iife.min.js', 'media/webview/compare.js'],
    styles: ['media/vendor/uPlot.min.css', 'media/webview/compare.css'],
    lang: getLanguage(),
  });
  panel.webview.onDidReceiveMessage((m: { type: string; file?: number }) => {
    if (m.type === 'file' && m.file !== undefined && files[m.file]) {
      const rel = files[m.file];
      const ua = vscode.Uri.file(path.join(a.dir, 'code', ...rel.split('/')));
      const ub = vscode.Uri.file(path.join(b.dir, 'code', ...rel.split('/')));
      void vscode.commands.executeCommand('vscode.diff', ua, ub, `${rel}: ${nameA} ↔ ${nameB}`);
    }
  });
  return panel;
}

export interface F5Api {
  tree: RunsTreeProvider;
  testHooks: typeof f5TestHooks;
  compare(a: RunData, b: RunData): vscode.WebviewPanel;
}

/** F5: experiment list, comparison, star, restore, delete. */
export function registerF5(ctx: vscode.ExtensionContext, watcher: RunWatcher): F5Api {
  const tree = new RunsTreeProvider(watcher);
  const view = vscode.window.createTreeView('deepfriend.runs', { treeDataProvider: tree, canSelectMany: true, showCollapseAll: false });
  ctx.subscriptions.push(view, watcher.onDidUpdate(() => tree.refresh()));

  const pickRun = async (title: string, exclude?: string): Promise<RunData | undefined> => {
    const items = tree
      .runs()
      .filter((r) => r.dir !== exclude)
      .map((r) => ({ label: RunItem.label(r.meta), description: new RunItem(r).description as string, run: r }));
    return (await vscode.window.showQuickPick(items, { title }))?.run;
  };

  ctx.subscriptions.push(
    vscode.commands.registerCommand('deepfriend.openRun', (item?: RunItem) => item && vscode.commands.executeCommand('deepfriend.openDoctor', item.run.dir)),
    vscode.commands.registerCommand('deepfriend.toggleStar', (item?: RunItem) => {
      if (!item) {
        return;
      }
      writeMeta(item.run.dir, { starred: !item.run.meta.starred });
      watcher.refresh(item.run.dir);
      tree.refresh();
    }),
    vscode.commands.registerCommand('deepfriend.restoreRun', async (item?: RunItem) => {
      const run = item?.run ?? (await pickRun(t('runs.pick.restore')));
      if (run) {
        await restoreRun(run);
      }
    }),
    vscode.commands.registerCommand('deepfriend.deleteRun', async (item?: RunItem) => {
      if (!item) {
        return;
      }
      if (!(await confirm(t('runs.delete.confirm', { name: RunItem.label(item.run.meta) }), t('runs.delete.yes')))) {
        return;
      }
      try {
        await vscode.workspace.fs.delete(vscode.Uri.file(item.run.dir), { recursive: true, useTrash: !f5TestHooks.autoConfirm });
      } catch {
        await vscode.workspace.fs.delete(vscode.Uri.file(item.run.dir), { recursive: true, useTrash: false });
      }
      watcher.store.forget(item.run.dir);
      tree.refresh();
    }),
    vscode.commands.registerCommand('deepfriend.compareRuns', async (item?: RunItem, selected?: RunItem[]) => {
      let a: RunData | undefined;
      let b: RunData | undefined;
      const sel = selected ?? [...view.selection];
      if (sel.length === 2) {
        [a, b] = [sel[0].run, sel[1].run];
      } else {
        a = item?.run ?? (await pickRun(t('runs.pick.first')));
        b = a && (await pickRun(t('runs.pick.second'), a.dir));
      }
      if (a && b) {
        comparePanel(ctx, a, b);
      }
    }),
  );
  return { tree, testHooks: f5TestHooks, compare: (a, b) => comparePanel(ctx, a, b) };
}
