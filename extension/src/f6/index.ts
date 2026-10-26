import { execFile, spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vscode from 'vscode';
import type { AnalysisStore } from '../f1/analysisStore';
import { getLanguage, t } from '../i18n/i18n';
import { LineSplitter } from '../server/protocol';

export interface ExportRequest {
  framework: 'pytorch' | 'keras';
  modelPath: string; // .py file (PyTorch) or .keras/.h5 (Keras)
  className?: string;
  weights?: string;
  input?: string; // "1,3,224,224"
  outDir: string;
}

export interface ExportResult {
  ok: boolean;
  tflite?: string;
  flutterDir?: string;
  inputShape?: number[];
  outputShape?: number[];
  verify?: { maxAbsDiff: number | null; cosine: number | null; passed: boolean };
  colab?: string;
  error?: { code: string; message: string; package?: string };
}

export interface F6Deps {
  python(): Promise<string[] | null>;
  bundledPath: string;
  output: vscode.OutputChannel;
}

export const f6TestHooks = { autoConfirm: false };

function exportEnvPython(root: string): string {
  return process.platform === 'win32' ? path.join(root, '.deepfriend', 'export-env', 'Scripts', 'python.exe') : path.join(root, '.deepfriend', 'export-env', 'bin', 'python');
}

function canImport(python: string[], module: string): Promise<boolean> {
  return new Promise((resolve) => {
    execFile(python[0], [...python.slice(1), '-c', `import ${module}`], { timeout: 120_000, windowsHide: true }, (err) => resolve(!err));
  });
}

function run(python: string[], args: string[], env: NodeJS.ProcessEnv, output: vscode.OutputChannel, onProgress?: (step: string) => void, token?: vscode.CancellationToken): Promise<{ code: number; lines: string[] }> {
  return new Promise((resolve) => {
    const p = spawn(python[0], [...python.slice(1), ...args], { env, windowsHide: true });
    const lines: string[] = [];
    const out = new LineSplitter();
    const err = new LineSplitter();
    token?.onCancellationRequested(() => p.kill());
    p.stdout.on('data', (b: Buffer) => {
      for (const l of out.push(b)) {
        lines.push(l);
        output.appendLine(l);
        if (l.startsWith('DF_PROGRESS ')) {
          onProgress?.(l.slice('DF_PROGRESS '.length));
        }
      }
    });
    p.stderr.on('data', (b: Buffer) => err.push(b).forEach((l) => output.appendLine(l)));
    p.on('error', (e) => {
      output.appendLine(String(e));
      resolve({ code: -1, lines });
    });
    p.on('close', (code) => resolve({ code: code ?? -1, lines }));
  });
}

/** Run the export CLI; on a missing converter for PyTorch, offer a private export environment. */
export async function runExport(req: ExportRequest, deps: F6Deps, token?: vscode.CancellationToken, report?: (msg: string) => void): Promise<ExportResult> {
  const base = await deps.python();
  if (!base) {
    return { ok: false, error: { code: 'no_python', message: t('err.noPython') } };
  }
  const root = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? path.dirname(req.modelPath);
  let python = base;
  if (req.framework === 'pytorch') {
    const envPy = exportEnvPython(root);
    if (fs.existsSync(envPy)) {
      python = [envPy];
    } else if (!(await canImport(base, 'litert_torch'))) {
      const yes = t('export.install.yes');
      const ok = f6TestHooks.autoConfirm || (await vscode.window.showInformationMessage(t('export.install.ask'), { modal: true }, yes)) === yes;
      if (ok) {
        report?.(t('export.install.progress'));
        const venv = await run(base, ['-m', 'venv', '--system-site-packages', path.join(root, '.deepfriend', 'export-env')], process.env, deps.output, undefined, token);
        if (venv.code === 0) {
          const pip = await run([envPy], ['-m', 'pip', 'install', 'litert-torch', 'ai-edge-litert'], process.env, deps.output, undefined, token);
          if (pip.code === 0) {
            python = [envPy];
          }
        }
      }
    }
  }
  const args = ['-m', 'deepfriend.export.cli', '--out', req.outDir, '--lang', getLanguage()];
  if (req.framework === 'keras') {
    args.push('--keras', req.modelPath);
  } else {
    args.push('--torch-file', req.modelPath, '--class', req.className ?? '', '--input', req.input ?? '');
    if (req.weights) {
      args.push('--weights', req.weights);
    }
  }
  const env = { ...process.env, PYTHONPATH: deps.bundledPath + (process.env.PYTHONPATH ? path.delimiter + process.env.PYTHONPATH : ''), PYTHONIOENCODING: 'utf-8', TF_CPP_MIN_LOG_LEVEL: '3' };
  deps.output.appendLine(`> ${python.join(' ')} ${args.join(' ')}`);
  const res = await run(python, args, env, deps.output, (step) => report?.(t(`export.step.${step}` as never)), token);
  const line = [...res.lines].reverse().find((l) => l.startsWith('DEEPFRIEND_RESULT '));
  if (!line) {
    return { ok: false, error: { code: 'convert_failed', message: t('export.noResult') } };
  }
  return JSON.parse(line.slice('DEEPFRIEND_RESULT '.length)) as ExportResult;
}

export interface F6Api {
  runExport(req: ExportRequest): Promise<ExportResult>;
  testHooks: typeof f6TestHooks;
}

/** F6: "Export to app" command. */
export function registerF6(ctx: vscode.ExtensionContext, store: AnalysisStore, deps: F6Deps): F6Api {
  const show = async (r: ExportResult): Promise<void> => {
    if (r.ok) {
      const open = t('export.openFolder');
      const readme = t('export.openReadme');
      const passed = r.verify?.passed ? t('export.verifyPassed') : t('export.verifyFailed', { diff: String(r.verify?.maxAbsDiff ?? '?') });
      const pick = await vscode.window.showInformationMessage(t('export.done', { file: path.basename(r.tflite ?? ''), verify: passed }), open, readme);
      if (pick === open && r.tflite) {
        await vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(r.tflite));
      } else if (pick === readme && r.flutterDir) {
        await vscode.window.showTextDocument(vscode.Uri.file(path.join(r.flutterDir, 'README.md')));
      }
      return;
    }
    const code = r.error?.code ?? 'convert_failed';
    if (code === 'platform_unsupported' && r.colab) {
      const open = t('export.openColab');
      if ((await vscode.window.showWarningMessage(t('export.platform'), open)) === open) {
        await vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(r.colab));
      }
      return;
    }
    if (code === 'converter_missing' && r.error?.package) {
      const install = t('err.install');
      if ((await vscode.window.showWarningMessage(t('export.missing', { pkg: r.error.package }), install)) === install) {
        await vscode.commands.executeCommand('deepfriend.installModule', r.error.package);
      }
      return;
    }
    const log = t('status.pick.log').replace(/^\$\([^)]*\)\s*/, '');
    if ((await vscode.window.showErrorMessage(t('export.failed', { message: r.error?.message ?? code }), log)) === log) {
      deps.output.show();
    }
  };

  const withProgress = (req: ExportRequest) =>
    vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: t('export.title'), cancellable: true }, (progress, token) =>
      runExport(req, deps, token, (message) => progress.report({ message })),
    );

  ctx.subscriptions.push(
    vscode.commands.registerCommand('deepfriend.exportToApp', async () => {
      const kind = await vscode.window.showQuickPick(
        [
          { label: t('export.pick.torch'), id: 'pytorch' as const },
          { label: t('export.pick.keras'), id: 'keras' as const },
        ],
        { title: t('export.title') },
      );
      if (!kind) {
        return;
      }
      const root = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
      const outDir = path.join(root ?? path.dirname(vscode.window.activeTextEditor?.document.fileName ?? '.'), 'deepfriend_export');
      let req: ExportRequest;
      if (kind.id === 'keras') {
        const file = await vscode.window.showOpenDialog({ title: t('export.pick.kerasFile'), filters: { Keras: ['keras', 'h5'] }, canSelectMany: false });
        if (!file?.[0]) {
          return;
        }
        req = { framework: 'keras', modelPath: file[0].fsPath, outDir };
      } else {
        const doc = vscode.window.activeTextEditor?.document;
        const models = doc ? (store.get(doc.uri.toString())?.models ?? []).filter((m) => m.kind === 'torch') : [];
        if (!doc || !models.length) {
          void vscode.window.showWarningMessage(t('export.noModel'));
          return;
        }
        const m = models.length === 1 ? models[0] : (await vscode.window.showQuickPick(models.map((x) => ({ label: x.name, model: x })), { title: t('export.pick.class') }))?.model;
        if (!m) {
          return;
        }
        const w = await vscode.window.showQuickPick([{ label: t('export.pick.weightsFile'), id: 'file' }, { label: t('export.pick.noWeights'), id: 'none' }], { title: t('export.pick.weights') });
        if (!w) {
          return;
        }
        let weights: string | undefined;
        if (w.id === 'file') {
          const f = await vscode.window.showOpenDialog({ title: t('export.pick.weights'), filters: { PyTorch: ['pt', 'pth', 'bin'] }, canSelectMany: false });
          if (!f?.[0]) {
            return;
          }
          weights = f[0].fsPath;
        }
        const suggested = m.input?.shapes[0]?.replace(/^\(\s*[^,]+/, '(1') ?? '(1, 3, 224, 224)';
        const input = await vscode.window.showInputBox({ title: t('export.pick.input'), value: suggested.replace(/[()\s]/g, ''), validateInput: (v) => (/^\s*\d+(\s*,\s*\d+)*\s*$/.test(v) ? null : t('input.invalid')) });
        if (!input) {
          return;
        }
        if (doc.isDirty) {
          await doc.save();
        }
        req = { framework: 'pytorch', modelPath: doc.uri.fsPath, className: m.name, weights, input, outDir };
      }
      await show(await withProgress(req));
    }),
  );
  return { runExport: (req) => runExport(req, deps), testHooks: f6TestHooks };
}
