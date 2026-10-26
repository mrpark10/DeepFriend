import { spawn } from 'node:child_process';
import * as vscode from 'vscode';
import { config, detectFramework } from './config';
import { AnalysisStore } from './f1/analysisStore';
import { registerF1 } from './f1';
import { registerF7 } from './f7';
import { ModelView, registerF4 } from './f4';
import { F3Api, registerF3 } from './f3';
import { F2Api, registerF2 } from './f2';
import { registerAi } from './ai';
import * as fs from 'node:fs';
import { setLanguage, t } from './i18n/i18n';
import { findPython, PythonInfo } from './python/findPython';
import { nodeFindDeps } from './python/nodeDeps';
import { pythonExtensionInterpreter } from './python/pythonExtension';
import { openIssue } from './report';
import { AnalysisClient, ProcessLike, Status } from './server/client';
import type { AnalysisResult } from './server/types';
import { StatusBar } from './status';
import { applyTerminalEnv } from './terminalEnv';

export interface DeepFriendApi {
  client: AnalysisClient;
  store: AnalysisStore;
  errors: F3Api;
  doctor: F2Api;
  modelView: ModelView;
  status(): Status;
  python(): PythonInfo | null;
  ensureServer(): Promise<boolean>;
}

const nodeSpawner = (cmd: string, args: string[], env: NodeJS.ProcessEnv): ProcessLike =>
  spawn(cmd, args, { env, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] }) as unknown as ProcessLike;

export async function activate(ctx: vscode.ExtensionContext): Promise<DeepFriendApi> {
  setLanguage(config.language());
  const log = vscode.window.createOutputChannel('DeepFriend');
  const status = new StatusBar();
  const client = new AnalysisClient(nodeSpawner, (line) => log.appendLine(line));
  const bundled = ctx.asAbsolutePath('python');
  let python: PythonInfo | null = null;
  let starting: Promise<boolean> | null = null;
  ctx.subscriptions.push(log, status, { dispose: () => client.dispose() });

  applyTerminalEnv(ctx, bundled);

  client.onStatus((s) => {
    status.set(s);
    if (s === 'error') {
      void vscode.window.showErrorMessage(t('server.crashedGiveUp'), t('server.restart'), t('server.report')).then((pick) => {
        if (pick === t('server.restart')) {
          void client.restart().catch(() => undefined);
        } else if (pick === t('server.report')) {
          void openIssue(ctx, client.info);
        }
      });
    }
  });

  const resolvePython = async (): Promise<PythonInfo | null> => {
    const extPath = await pythonExtensionInterpreter(() => void restartWithNewPython(), ctx.subscriptions);
    return findPython({ setting: config.pythonPath(), pythonExtPath: extPath }, nodeFindDeps);
  };

  /** Start the analysis server once, lazily (only when a deep-learning file is opened). */
  const ensureServer = (): Promise<boolean> => {
    if (client.status === 'ready' || client.status === 'busy') {
      return Promise.resolve(true);
    }
    if (starting) {
      return starting;
    }
    starting = (async () => {
      python = await resolvePython();
      if (!python) {
        status.set('noPython');
        log.appendLine('No suitable Python found');
        void vscode.window.showWarningMessage(t('python.notFound'), t('python.install'), t('python.choose')).then((pick) => {
          if (pick === t('python.install')) {
            void vscode.env.openExternal(vscode.Uri.parse('https://www.python.org/downloads/'));
          } else if (pick === t('python.choose')) {
            void vscode.commands.executeCommand('deepfriend.selectPython');
          }
        });
        return false;
      }
      log.appendLine(`Using Python ${python.version} at ${python.command.join(' ')} (${python.source})`);
      try {
        await client.start(python, bundled);
        return true;
      } catch (e) {
        log.appendLine('Failed to start analysis server: ' + (e as Error).message);
        status.set('error');
        return false;
      }
    })().finally(() => {
      starting = null;
    });
    return starting;
  };

  const restartWithNewPython = async (): Promise<void> => {
    if (!client.info) {
      return; // server not started yet; the next ensureServer() picks up the new interpreter
    }
    const next = await resolvePython();
    if (next) {
      python = next;
      log.appendLine(`Switching to Python ${next.version} at ${next.command.join(' ')}`);
      await client.switchPython(next).catch((e) => log.appendLine('switch failed: ' + (e as Error).message));
    }
  };

  const updateFramework = (editor: vscode.TextEditor | undefined): void => {
    const doc = editor?.document;
    const fw = doc && doc.languageId === 'python' ? detectFramework(doc.getText()) : null;
    status.setFramework(fw);
    if (fw) {
      void ensureServer();
    }
  };

  ctx.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(updateFramework),
    vscode.workspace.onDidSaveTextDocument((doc) => {
      if (doc === vscode.window.activeTextEditor?.document) {
        updateFramework(vscode.window.activeTextEditor);
      }
    }),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration('deepfriend.language')) {
        setLanguage(config.language());
        status.render();
      }
      if (e.affectsConfiguration('deepfriend.pythonPath')) {
        void restartWithNewPython();
      }
      if (e.affectsConfiguration('deepfriend.autoRecord')) {
        applyTerminalEnv(ctx, bundled);
      }
    }),
    vscode.commands.registerCommand('deepfriend.restartServer', async () => {
      if (!python) {
        await ensureServer();
        return;
      }
      await client.restart().catch((e) => log.appendLine('restart failed: ' + (e as Error).message));
    }),
    vscode.commands.registerCommand('deepfriend.selectPython', async () => {
      const picked = await vscode.window.showOpenDialog({ canSelectMany: false, title: t('python.chooseTitle'), openLabel: t('common.ok') });
      if (picked?.[0]) {
        await config.set('pythonPath', picked[0].fsPath);
      }
    }),
    vscode.commands.registerCommand('deepfriend.reportIssue', () => openIssue(ctx, client.info)),
    vscode.commands.registerCommand('deepfriend.showStatus', async () => {
      const info = client.info;
      const items: (vscode.QuickPickItem & { id?: string })[] = [];
      if (python) {
        items.push({ label: t('status.pick.python', { path: python.command.join(' '), version: python.version }) });
      }
      if (info) {
        items.push({
          label: t('status.pick.frameworks', {
            torch: info.torch ?? t('status.pick.notInstalled'),
            tensorflow: info.tensorflow ?? t('status.pick.notInstalled'),
          }),
        });
      }
      items.push(
        { label: '', kind: vscode.QuickPickItemKind.Separator },
        { label: t('status.pick.restart'), id: 'restart' },
        { label: t('status.pick.selectPython'), id: 'python' },
        { label: t('status.pick.log'), id: 'log' },
        { label: t('status.pick.report'), id: 'report' },
      );
      const pick = await vscode.window.showQuickPick(items, { title: 'DeepFriend' });
      if (pick?.id === 'restart') {
        await vscode.commands.executeCommand('deepfriend.restartServer');
      } else if (pick?.id === 'python') {
        await vscode.commands.executeCommand('deepfriend.selectPython');
      } else if (pick?.id === 'log') {
        log.show();
      } else if (pick?.id === 'report') {
        await vscode.commands.executeCommand('deepfriend.reportIssue');
      }
    }),
  );

  // ---- F1: shape analysis ----
  let inflight = 0;
  const store = new AnalysisStore(async (uri, text) => {
    if (!(await ensureServer())) {
      return null;
    }
    inflight++;
    status.set('busy');
    try {
      return await client.request<AnalysisResult | null>('analyze', { uri, text, runtime: config.runtimeCheck() }, 60_000);
    } finally {
      inflight--;
      if (inflight === 0 && client.status !== 'error') {
        status.set(client.status === 'busy' ? 'ready' : client.status);
      }
    }
  });
  const f1 = registerF1(ctx, store);
  registerF7(ctx, store);
  const modelView = registerF4(ctx, store);
  const ai = registerAi(ctx);
  const errors = registerF3(ctx, {
    pythonCommand: () => python?.command ?? null,
    aiAvailable: () => ai.available(),
    askAi: async (reported) => {
      const f = reported.error.frames[reported.error.frames.length - 1];
      let code = '';
      try {
        if (f && fs.existsSync(f.file)) {
          const lines = fs.readFileSync(f.file, 'utf8').split('\n');
          code = lines.slice(Math.max(0, f.line - 6), f.line + 4).join('\n');
        }
      } catch {
        // unreadable file: ask without code
      }
      return ai.explainError(reported.error, code);
    },
  });
  const doctor = registerF2(
    ctx,
    {
      findArgument: async (text, kind) => ((await ensureServer()) ? client.request('findArgument', { text, kind }) : null),
      polish: (f, text) => ai.polishDoctor(f.id, text),
    },
    (oldV) => {
      const yes = t('doctor.update.yes');
      void vscode.window
        .showInformationMessage(t('doctor.update.python', { old: oldV, new: String(ctx.extension.packageJSON.version) }), yes)
        .then((p) => p === yes && vscode.commands.executeCommand('deepfriend.installModule', 'deepfriend', true));
    },
  );
  ctx.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument((d) => store.onOpen(d)),
    vscode.workspace.onDidChangeTextDocument((e) => {
      if (e.contentChanges.length) {
        store.onChange(e.document);
      }
    }),
    vscode.workspace.onDidSaveTextDocument((d) => store.onSave(d)),
    vscode.workspace.onDidCloseTextDocument((d) => store.onClose(d)),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration('deepfriend.language') || e.affectsConfiguration('deepfriend.shapeHints')) {
        f1.refresh();
      }
      if (e.affectsConfiguration('deepfriend.runtimeCheck')) {
        for (const d of vscode.workspace.textDocuments) {
          store.onSave(d);
        }
      }
    }),
  );
  for (const d of vscode.workspace.textDocuments) {
    store.onOpen(d);
  }

  updateFramework(vscode.window.activeTextEditor);

  return {
    client,
    store,
    errors,
    doctor,
    modelView,
    status: () => status.get(),
    python: () => python,
    ensureServer,
  };
}

export function deactivate(): void {
  // Disposables registered on the context stop the server.
}
