import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { getLanguage, t } from '../i18n/i18n';
import { escapeHtml, webviewHtml } from '../webview/html';
import { explain, Explanation, renderExplanation } from './errorDictionary';
import { ErrorHub, ReportedError, TerminalBuffer } from './errorHub';
import { findTracebacks, Frame, ParsedError, userFrame } from './traceback';

export interface F3Deps {
  pythonCommand(): string[] | null;
  /** Optional AI fallback for errors that are not in the dictionary (Task 16). */
  askAi?: (err: ReportedError) => Promise<{ meaning: string; cause: string; fix: string } | null>;
  aiAvailable?: () => boolean;
}

export interface F3Api {
  hub: ErrorHub;
  card: ErrorCard;
}

const IGNORED_TYPES = new Set(['KeyboardInterrupt', 'SystemExit', 'GeneratorExit']);

function workspaceFolders(): string[] {
  return (vscode.workspace.workspaceFolders ?? []).map((f) => f.uri.fsPath);
}

export class ErrorCard implements vscode.Disposable {
  private panel: vscode.WebviewPanel | undefined;
  private current: { reported: ReportedError; explanation: Explanation | null; frame?: Frame } | undefined;
  lastHtml = '';

  constructor(private extensionUri: vscode.Uri, private deps: F3Deps) {}

  show(reported: ReportedError, explanation: Explanation | null, ai?: { meaning: string; cause: string; fix: string }): void {
    const frame = userFrame(reported.error, workspaceFolders());
    this.current = { reported, explanation, frame };
    if (!this.panel) {
      this.panel = vscode.window.createWebviewPanel('deepfriend.errorCard', t('err.cardTitle'), { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true }, {
        enableScripts: true,
        localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, 'media')],
      });
      this.panel.onDidDispose(() => (this.panel = undefined));
      this.panel.webview.onDidReceiveMessage((m: { type: string }) => this.onAction(m.type));
    } else {
      this.panel.reveal(vscode.ViewColumn.Beside, true);
    }
    this.panel.title = t('err.cardTitle');
    this.lastHtml = this.render(reported, explanation, frame, ai);
    this.panel.webview.html = this.lastHtml;
  }

  private render(reported: ReportedError, x: Explanation | null, frame: Frame | undefined, ai?: { meaning: string; cause: string; fix: string }): string {
    const e = reported.error;
    const parts: string[] = [];
    const where = frame ? `<p class="muted">${escapeHtml(t('err.where', { file: path.basename(frame.file), line: frame.line }))}</p>` : '';
    if (x) {
      const r = renderExplanation(x);
      parts.push(`<div class="card"><h1>${escapeHtml(r.title)}</h1>${where}
<h2>${escapeHtml(t('err.meaning'))}</h2><p>${escapeHtml(r.meaning)}</p>
<h2>${escapeHtml(t('err.cause'))}</h2><p>${escapeHtml(r.cause)}</p>
<h2>${escapeHtml(t('err.fix'))}</h2><p>${escapeHtml(r.fix)}</p></div>`);
    } else if (ai) {
      parts.push(`<div class="card"><h1>${escapeHtml(e.type)}<span class="badge">${escapeHtml(t('ai.badge'))}</span></h1>${where}
<h2>${escapeHtml(t('err.meaning'))}</h2><p>${escapeHtml(ai.meaning)}</p>
<h2>${escapeHtml(t('err.cause'))}</h2><p>${escapeHtml(ai.cause)}</p>
<h2>${escapeHtml(t('err.fix'))}</h2><p>${escapeHtml(ai.fix)}</p></div>`);
    } else {
      parts.push(`<div class="card"><h1>${escapeHtml(t('err.unknownTitle', { type: e.type }))}</h1>${where}<p>${escapeHtml(t('err.unknownBody'))}</p></div>`);
    }
    const buttons: string[] = [];
    if (frame) {
      buttons.push(`<button data-action="goto">${escapeHtml(t('err.goto'))}</button>`);
    }
    if (x?.actions.includes('installModule')) {
      buttons.push(`<button data-action="install">${escapeHtml(t('err.install'))} (${escapeHtml(x.vars.pip)})</button>`);
    }
    if (!x && !ai && this.deps.askAi && this.deps.aiAvailable?.()) {
      buttons.push(`<button data-action="ai" class="secondary">${escapeHtml(t('err.askAi'))}</button>`);
    }
    buttons.push(`<button data-action="search" class="secondary">${escapeHtml(t('err.search'))}</button>`);
    parts.push(`<div class="actions">${buttons.join('')}</div>`);
    parts.push(`<h2>${escapeHtml(t('err.raw'))}</h2><pre>${escapeHtml(`${e.type}: ${e.message}`)}</pre>`);
    if (!this.panel) {
      return '';
    }
    return webviewHtml(this.panel.webview, this.extensionUri, { title: t('err.cardTitle'), body: parts.join('\n'), scripts: ['media/webview/card.js'], lang: getLanguage() });
  }

  private async onAction(type: string): Promise<void> {
    const cur = this.current;
    if (!cur) {
      return;
    }
    if (type === 'goto' && cur.frame) {
      await vscode.commands.executeCommand('deepfriend.goToErrorLine', cur.frame);
    } else if (type === 'install' && cur.explanation) {
      await vscode.commands.executeCommand('deepfriend.installModule', cur.explanation.vars.pip);
    } else if (type === 'search') {
      const q = `${cur.reported.error.type}: ${cur.reported.error.message.split('\n')[0]}`;
      await vscode.env.openExternal(vscode.Uri.parse('https://www.google.com/search?q=' + encodeURIComponent(q)));
    } else if (type === 'ai' && this.deps.askAi) {
      const ans = await this.deps.askAi(cur.reported);
      if (ans) {
        this.show(cur.reported, null, ans);
      }
    }
  }

  dispose(): void {
    this.panel?.dispose();
  }
}

async function goToFrame(frame: Frame): Promise<boolean> {
  const cell = /^Cell In\[(\d+)\]$/.exec(frame.file);
  if (cell) {
    const order = parseInt(cell[1], 10);
    for (const nb of vscode.workspace.notebookDocuments) {
      const c = nb.getCells().find((x) => x.executionSummary?.executionOrder === order);
      if (c) {
        const line = Math.max(0, Math.min(frame.line - 1, c.document.lineCount - 1));
        await vscode.window.showTextDocument(c.document, { selection: new vscode.Range(line, 0, line, 0) });
        return true;
      }
    }
    return false;
  }
  const file = path.isAbsolute(frame.file) ? frame.file : path.join(workspaceFolders()[0] ?? '', frame.file);
  if (!fs.existsSync(file)) {
    return false;
  }
  const doc = await vscode.workspace.openTextDocument(file);
  const line = Math.max(0, Math.min(frame.line - 1, doc.lineCount - 1));
  const range = doc.lineAt(line).range;
  await vscode.window.showTextDocument(doc, { selection: range, viewColumn: vscode.ViewColumn.One });
  return true;
}

/** F3: Korean explanations for Python errors from terminals, notebooks and watch() runs. */
export function registerF3(ctx: vscode.ExtensionContext, deps: F3Deps): F3Api {
  const hub = new ErrorHub();
  const card = new ErrorCard(ctx.extensionUri, deps);
  ctx.subscriptions.push(card);

  hub.onError(async (reported) => {
    if (IGNORED_TYPES.has(reported.error.type)) {
      return;
    }
    const x = explain(reported.error);
    const frame = userFrame(reported.error, workspaceFolders());
    const title = x ? renderExplanation(x).title : t('err.unknownTitle', { type: reported.error.type });
    const buttons = [t('err.details')];
    if (frame) {
      buttons.push(t('err.goto'));
    }
    if (x?.actions.includes('installModule')) {
      buttons.push(t('err.install'));
    }
    const pick = await vscode.window.showErrorMessage(`DeepFriend: ${title}`, ...buttons);
    if (pick === t('err.details')) {
      card.show(reported, x);
    } else if (pick === t('err.goto') && frame) {
      await vscode.commands.executeCommand('deepfriend.goToErrorLine', frame);
    } else if (pick === t('err.install') && x) {
      await vscode.commands.executeCommand('deepfriend.installModule', x.vars.pip);
    }
  });

  // 1) integrated terminal output (shell integration, VS Code 1.93+)
  const win = vscode.window as unknown as {
    onDidStartTerminalShellExecution?: vscode.Event<{ execution: { read(): AsyncIterable<string> } }>;
  };
  if (win.onDidStartTerminalShellExecution) {
    ctx.subscriptions.push(
      win.onDidStartTerminalShellExecution(async (e) => {
        const buf = new TerminalBuffer();
        try {
          for await (const chunk of e.execution.read()) {
            for (const err of buf.push(chunk)) {
              hub.report(err, 'terminal');
            }
          }
        } catch {
          // terminal closed; nothing to do
        }
      }),
    );
  }

  // 2) notebook cell error outputs
  const ERROR_MIME = 'application/vnd.code.notebook.error';
  ctx.subscriptions.push(
    vscode.workspace.onDidChangeNotebookDocument((e) => {
      for (const change of e.cellChanges) {
        for (const out of change.outputs ?? []) {
          for (const item of out.items) {
            if (item.mime !== ERROR_MIME) {
              continue;
            }
            try {
              const j = JSON.parse(new TextDecoder().decode(item.data)) as { name: string; message: string; stack?: string };
              const parsed = j.stack ? findTracebacks(j.stack)[0] : undefined;
              const err: ParsedError = { type: j.name, message: j.message, frames: parsed?.frames ?? [], raw: j.stack ?? '' };
              hub.report(err, 'notebook');
            } catch {
              // malformed output
            }
          }
        }
      }
    }),
  );

  // 3) errors recorded by deepfriend.watch() (works even without shell integration)
  const watcher = vscode.workspace.createFileSystemWatcher('**/.deepfriend/runs/*/error.json');
  const onRunError = async (uri: vscode.Uri) => {
    try {
      const j = JSON.parse(Buffer.from(await vscode.workspace.fs.readFile(uri)).toString('utf8')) as {
        type: string;
        message: string;
        frames?: Frame[];
        traceback?: string;
      };
      hub.report({ type: j.type, message: j.message, frames: j.frames ?? [], raw: j.traceback ?? '' }, 'runFile');
    } catch {
      // partially written file; the change event will fire again
    }
  };
  ctx.subscriptions.push(watcher, watcher.onDidCreate(onRunError), watcher.onDidChange(onRunError));

  ctx.subscriptions.push(
    vscode.commands.registerCommand('deepfriend.goToErrorLine', async (frame: Frame) => {
      if (!(await goToFrame(frame))) {
        void vscode.window.showWarningMessage(t('err.noLocation'));
      }
    }),
    vscode.commands.registerCommand('deepfriend.installModule', async (pkg: string, upgrade?: boolean) => {
      const py = deps.pythonCommand();
      if (!py) {
        void vscode.window.showWarningMessage(t('err.noPython'));
        return;
      }
      if (!/^[A-Za-z0-9._-]+$/.test(pkg)) {
        return; // never pass anything unexpected to pip
      }
      const task = new vscode.Task(
        { type: 'deepfriend', task: 'install' },
        vscode.TaskScope.Workspace,
        t('err.installTask', { pkg }),
        'DeepFriend',
        new vscode.ProcessExecution(py[0], [...py.slice(1), '-m', 'pip', 'install', ...(upgrade ? ['-U'] : []), pkg]),
      );
      await vscode.tasks.executeTask(task);
    }),
  );

  return { hub, card };
}
