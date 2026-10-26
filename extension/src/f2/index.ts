import * as path from 'node:path';
import * as vscode from 'vscode';
import { getLanguage, t } from '../i18n/i18n';
import { Emitter } from '../server/client';
import type { ArgLocation } from '../server/types';
import { escapeHtml, webviewHtml } from '../webview/html';
import { buildChart } from './chart';
import { diagnose, DoctorAction, Finding } from './doctorRules';
import { actionLabel, editForAction, formatNumber, renderFinding, snippet } from './doctorText';
import { RunData, RunStore } from './runStore';

export interface F2Deps {
  findArgument(text: string, kind: 'lr' | 'batch_size'): Promise<ArgLocation[] | null>;
  /** Optional AI rephrasing of a rendered finding (Task 16). */
  polish?: (f: Finding, text: { what: string; why: string; try: string }) => Promise<{ what: string; why: string; try: string } | null>;
}

export const testHooks = { autoConfirm: false };

const GLOB = '**/.deepfriend/{runs,live}/*/{meta.json,metrics.jsonl}';

/** Root of the project a run belongs to: <root>/.deepfriend/runs/<id>. */
export function projectRoot(runDir: string): string {
  return path.dirname(path.dirname(path.dirname(runDir)));
}

function versionLess(a: string, b: string): boolean {
  const pa = a.split('.').map((x) => parseInt(x, 10) || 0);
  const pb = b.split('.').map((x) => parseInt(x, 10) || 0);
  for (let i = 0; i < 3; i++) {
    if ((pa[i] ?? 0) !== (pb[i] ?? 0)) {
      return (pa[i] ?? 0) < (pb[i] ?? 0);
    }
  }
  return false;
}

/** Watches `.deepfriend` run folders in the workspace. */
export class RunWatcher implements vscode.Disposable {
  readonly store = new RunStore();
  private emitter = new Emitter<RunData>();
  readonly onDidUpdate = this.emitter.event;
  private disposables: vscode.Disposable[] = [];
  private timer: ReturnType<typeof setInterval>;
  pinned: string | undefined;

  constructor() {
    const w = vscode.workspace.createFileSystemWatcher(GLOB);
    const onFile = (uri: vscode.Uri) => this.refresh(path.dirname(uri.fsPath));
    this.disposables.push(w, w.onDidCreate(onFile), w.onDidChange(onFile), w.onDidDelete((uri) => {
      if (uri.fsPath.endsWith('meta.json')) {
        this.store.forget(path.dirname(uri.fsPath));
      }
    }));
    this.timer = setInterval(() => {
      for (const r of this.store.all()) {
        if (r.meta.status === 'running') {
          this.refresh(r.dir);
        }
      }
    }, 1000);
    void this.scan();
  }

  async scan(): Promise<void> {
    const files = await vscode.workspace.findFiles('**/.deepfriend/{runs,live}/*/meta.json', '**/node_modules/**', 500);
    for (const f of files) {
      this.refresh(path.dirname(f.fsPath));
    }
  }

  refresh(dir: string): void {
    const before = this.store.get(dir);
    const prevLen = before ? before.series.train.length + before.series.epochs.length : -1;
    const prevMeta = before ? JSON.stringify(before.meta) : '';
    const r = this.store.refresh(dir);
    if (r && (r.series.train.length + r.series.epochs.length !== prevLen || JSON.stringify(r.meta) !== prevMeta)) {
      this.emitter.fire(r);
    }
  }

  /** The run the doctor shows: pinned one, else the newest. */
  current(): RunData | undefined {
    if (this.pinned) {
      const p = this.store.get(this.pinned);
      if (p) {
        return p;
      }
    }
    return this.store.all().sort((a, b) => (a.meta.startedAt < b.meta.startedAt ? 1 : -1))[0];
  }

  dispose(): void {
    clearInterval(this.timer);
    this.disposables.forEach((d) => d.dispose());
  }
}

interface PostedState {
  type: 'state';
  L: Record<string, string>;
  run: null | { title: string; status: string; meta: string; dir: string };
  chart: ReturnType<typeof buildChart>;
  findings: { id: string; what: string; why: string; try: string; ai: boolean; actions: string[] }[];
  note: string;
}

export class DoctorView implements vscode.WebviewViewProvider {
  private view: vscode.WebviewView | undefined;
  private findings: Finding[] = [];
  private polished = new Map<string, { what: string; why: string; try: string }>();
  lastMessage: PostedState | undefined;
  onAction?: (f: Finding, a: DoctorAction, run: RunData) => void;

  constructor(private extensionUri: vscode.Uri, private watcher: RunWatcher, private deps: F2Deps) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = { enableScripts: true, localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, 'media')] };
    view.webview.html = webviewHtml(view.webview, this.extensionUri, {
      title: t('doctor.title'),
      body: '<div id="root"></div>',
      scripts: ['media/vendor/uPlot.iife.min.js', 'media/webview/doctor.js'],
      styles: ['media/vendor/uPlot.min.css', 'media/webview/doctor.css'],
      lang: getLanguage(),
    });
    view.webview.onDidReceiveMessage((m: { type: string; finding?: number; action?: number }) => {
      if (m.type === 'ready') {
        this.update();
      } else if (m.type === 'copy') {
        void vscode.env.clipboard.writeText('import deepfriend; deepfriend.watch()');
      } else if (m.type === 'action' && m.finding !== undefined && m.action !== undefined) {
        const run = this.watcher.current();
        const f = this.findings[m.finding];
        const a = f?.actions[m.action];
        if (run && f && a && this.onAction) {
          this.onAction(f, a, run);
        }
      }
    });
    view.onDidChangeVisibility(() => view.visible && this.update());
  }

  update(): void {
    const run = this.watcher.current();
    const L = {
      what: t('doctor.what'), why: t('doctor.why'), try: t('doctor.try'), loss: t('doctor.loss'), acc: t('doctor.acc'),
      train: t('doctor.train'), val: t('doctor.val'), step: t('doctor.step'), copy: t('doctor.empty.copy'), copied: t('doctor.empty.copied'),
      emptyTitle: t('doctor.empty.title'), emptyBody: t('doctor.empty.body'), ai: t('ai.badge'),
    };
    let msg: PostedState;
    if (!run) {
      this.findings = [];
      msg = { type: 'state', L, run: null, chart: buildChart({ train: [], epochs: [], numClasses: null }), findings: [], note: '' };
    } else {
      this.findings = diagnose(run.series);
      const cfg = run.meta.config as Record<string, unknown>;
      const lastTrain = run.series.train[run.series.train.length - 1];
      const lr = typeof cfg.lr === 'number' ? formatNumber(cfg.lr) : '-';
      const enough = run.series.train.length >= 50 || run.series.epochs.length >= 3;
      msg = {
        type: 'state',
        L,
        run: {
          title: run.meta.name || run.meta.script || run.meta.id,
          status: t(`doctor.status.${run.meta.status}` as never),
          meta: t('doctor.meta', { epoch: (lastTrain?.epoch ?? run.series.epochs.length - 1) + 1, step: lastTrain?.step ?? 0, lr, batch: String(cfg.batch_size ?? '-') }),
          dir: run.dir,
        },
        chart: buildChart(run.series),
        findings: this.findings.map((f) => {
          const base = renderFinding(f);
          const key = run.dir + '|' + f.id + '|' + getLanguage();
          const ai = this.polished.get(key);
          if (!ai && this.deps.polish && !this.polished.has(key + '|pending')) {
            this.polished.set(key + '|pending', base);
            void this.deps.polish(f, base).then((p) => {
              if (p) {
                this.polished.set(key, p);
                this.update();
              }
            });
          }
          return { id: f.id, ...(ai ?? base), ai: !!ai, actions: f.actions.map(actionLabel) };
        }),
        note: enough ? t('doctor.allGood') : t('doctor.waiting'),
      };
    }
    this.lastMessage = msg;
    void this.view?.webview.postMessage(msg);
  }
}

/** Virtual documents for the "proposed change" diff. */
class PreviewProvider implements vscode.TextDocumentContentProvider {
  private contents = new Map<string, string>();
  set(uri: vscode.Uri, text: string): void {
    this.contents.set(uri.toString(), text);
  }
  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.contents.get(uri.toString()) ?? '';
  }
}

async function applyArgumentChange(deps: F2Deps, preview: PreviewProvider, run: RunData, action: DoctorAction): Promise<boolean> {
  if (action.kind !== 'setLr' && action.kind !== 'setBatch') {
    return false;
  }
  if (!run.meta.script) {
    void vscode.window.showWarningMessage(t('doctor.apply.noScript'));
    return false;
  }
  const file = path.join(projectRoot(run.dir), ...run.meta.script.split('/'));
  let doc: vscode.TextDocument;
  try {
    doc = await vscode.workspace.openTextDocument(file);
  } catch {
    void vscode.window.showWarningMessage(t('doctor.apply.noScript'));
    return false;
  }
  const kind = action.kind === 'setLr' ? 'lr' : 'batch_size';
  const locs = (await deps.findArgument(doc.getText(), kind)) ?? [];
  const wanted = (run.meta.config as Record<string, unknown>)[kind === 'lr' ? 'lr' : 'batch_size'];
  const loc = locs.find((l) => l.value !== null && l.value === wanted) ?? locs[0];
  const name = kind === 'lr' ? 'learning rate' : 'batch_size';
  if (!loc) {
    const hint = kind === 'lr' ? `lr=${typeof wanted === 'number' ? formatNumber(wanted * (action as { factor: number }).factor) : '...'}` : `batch_size=...`;
    void vscode.window.showWarningMessage(t('doctor.apply.notFound', { file: path.basename(file), name, hint }));
    return false;
  }
  const edit = editForAction(loc, action);
  if (!edit) {
    void vscode.window.showWarningMessage(t('doctor.apply.unknownValue', { name: loc.name }));
    return false;
  }
  const range = new vscode.Range(edit.line, edit.col, edit.line, edit.endCol);
  const before = doc.getText();
  const after = before.slice(0, doc.offsetAt(range.start)) + edit.text + before.slice(doc.offsetAt(range.end));
  const previewUri = vscode.Uri.parse(`deepfriend-preview:/${encodeURIComponent(path.basename(file))}?${Date.now()}`);
  preview.set(previewUri, after);
  if (!testHooks.autoConfirm) {
    await vscode.commands.executeCommand('vscode.diff', doc.uri, previewUri, `${path.basename(file)} ↔ DeepFriend`, { preview: true });
  }
  const oldText = loc.insert ? '(default)' : doc.getText(range);
  const newText = edit.text.replace(/^,\s*/, '').replace(/^[a-z_]+=/, '');
  const yes = t('doctor.apply.yes');
  const ok = testHooks.autoConfirm || (await vscode.window.showInformationMessage(t('doctor.apply.confirm', { file: path.basename(file), name: loc.name, old: oldText, new: newText }), { modal: true }, yes)) === yes;
  if (!ok) {
    return false;
  }
  const we = new vscode.WorkspaceEdit();
  we.replace(doc.uri, range, edit.text);
  await vscode.workspace.applyEdit(we);
  await doc.save();
  return true;
}

function showSamples(ctx: vscode.ExtensionContext, run: RunData): vscode.WebviewPanel {
  const panel = vscode.window.createWebviewPanel('deepfriend.samples', t('doctor.samples.title'), vscode.ViewColumn.Beside, {
    enableScripts: true,
    localResourceRoots: [vscode.Uri.joinPath(ctx.extensionUri, 'media')],
  });
  const items = run.samples?.items ?? [];
  const json = JSON.stringify({ items, labelText: t('doctor.samples.label') }).replace(/</g, '\\u003c');
  const body = items.length
    ? `<h1>${escapeHtml(t('doctor.samples.title'))}</h1><div id="grid" class="samples"></div><script type="application/json" id="samples-data">${json}</script>`
    : `<p>${escapeHtml(t('doctor.samples.none'))}</p><script type="application/json" id="samples-data">{"items":[],"labelText":""}</script><div id="grid"></div>`;
  panel.webview.html = webviewHtml(panel.webview, ctx.extensionUri, { title: t('doctor.samples.title'), body, scripts: ['media/webview/samples.js'], styles: ['media/webview/samples.css'], lang: getLanguage() });
  return panel;
}

export interface F2Api {
  watcher: RunWatcher;
  view: DoctorView;
  testHooks: typeof testHooks;
  applyAction(f: Finding, a: DoctorAction, run: RunData): Promise<boolean>;
}

/** F2: training doctor panel. */
export function registerF2(ctx: vscode.ExtensionContext, deps: F2Deps, onOutdatedPackage: (oldV: string) => void): F2Api {
  const watcher = new RunWatcher();
  const view = new DoctorView(ctx.extensionUri, watcher, deps);
  const preview = new PreviewProvider();
  ctx.subscriptions.push(
    watcher,
    vscode.window.registerWebviewViewProvider('deepfriend.doctor', view, { webviewOptions: { retainContextWhenHidden: true } }),
    vscode.workspace.registerTextDocumentContentProvider('deepfriend-preview', preview),
  );
  const bundled = String(ctx.extension.packageJSON.version ?? '0.0.0');
  let warnedVersion = false;
  watcher.onDidUpdate((r) => {
    view.update();
    const v = r.meta.deepfriendVersion;
    if (!warnedVersion && v && versionLess(v, bundled)) {
      warnedVersion = true;
      onOutdatedPackage(v);
    }
  });

  const applyAction = async (f: Finding, a: DoctorAction, run: RunData): Promise<boolean> => {
    if (a.kind === 'setLr' || a.kind === 'setBatch') {
      return applyArgumentChange(deps, preview, run, a);
    }
    if (a.kind === 'snippet') {
      const doc = await vscode.workspace.openTextDocument({ language: 'python', content: snippet(a.id, run.meta.framework) });
      await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
      return true;
    }
    if (a.kind === 'showSamples') {
      showSamples(ctx, run);
      return true;
    }
    return false;
  };
  view.onAction = (f, a, run) => void applyAction(f, a, run);

  ctx.subscriptions.push(
    vscode.commands.registerCommand('deepfriend.openDoctor', async (dir?: string) => {
      if (dir) {
        watcher.pinned = dir;
        watcher.refresh(dir);
      }
      await vscode.commands.executeCommand('deepfriend.doctor.focus');
      view.update();
    }),
  );
  return { watcher, view, applyAction, testHooks };
}
