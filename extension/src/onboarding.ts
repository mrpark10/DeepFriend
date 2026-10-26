import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import * as vscode from 'vscode';
import { t } from './i18n/i18n';

const PENDING = 'deepfriend.pendingExampleFile';
const WELCOMED = 'deepfriend.welcomed';

export const EXAMPLES = [
  { file: '01_mnist_mlp.py', key: 'example.01' },
  { file: '02_cifar_cnn.py', key: 'example.02' },
  { file: '03_keras_cnn.py', key: 'example.03' },
  { file: '04_shape_error.py', key: 'example.04' },
  { file: '05_doctor/nan.py', key: 'example.05' },
] as const;

function freeDir(base: string): string {
  let dir = base;
  for (let i = 2; fs.existsSync(dir); i++) {
    dir = `${base}-${i}`;
  }
  return dir;
}

/** Copy the bundled examples into `target` and return the folder that was created. */
export function copyExamples(extensionPath: string, target: string): string {
  const dest = freeDir(target);
  fs.cpSync(path.join(extensionPath, 'examples'), dest, { recursive: true, filter: (src) => !src.includes('__pycache__') });
  return dest;
}

export function registerOnboarding(ctx: vscode.ExtensionContext): void {
  ctx.subscriptions.push(
    vscode.commands.registerCommand('deepfriend.openExample', async (file?: string, targetDir?: string, open = true) => {
      const pick = file
        ? EXAMPLES.find((e) => e.file === file)
        : await vscode.window.showQuickPick(
            EXAMPLES.map((e) => ({ label: e.file, description: t(e.key as never), e })),
            { title: t('example.pickTitle') },
          ).then((p) => p?.e);
      if (!pick) {
        return undefined;
      }
      const dest = copyExamples(ctx.extensionPath, targetDir ?? path.join(os.homedir(), 'DeepFriend-examples'));
      const fileUri = vscode.Uri.file(path.join(dest, ...pick.file.split('/')));
      if (!open) {
        return dest;
      }
      if (vscode.workspace.workspaceFolders?.length) {
        await ctx.globalState.update(PENDING, fileUri.fsPath);
        await vscode.commands.executeCommand('vscode.openFolder', vscode.Uri.file(dest), { forceNewWindow: true });
      } else {
        await ctx.globalState.update(PENDING, fileUri.fsPath);
        await vscode.commands.executeCommand('vscode.openFolder', vscode.Uri.file(dest), { forceNewWindow: false });
      }
      return dest;
    }),
  );

  // A new window opened by "open example": show the chosen file.
  const pending = ctx.globalState.get<string>(PENDING);
  const folder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (pending && folder && pending.startsWith(folder) && fs.existsSync(pending)) {
    void ctx.globalState.update(PENDING, undefined);
    void vscode.window.showTextDocument(vscode.Uri.file(pending));
  }

  if (!ctx.globalState.get<boolean>(WELCOMED) && ctx.extensionMode !== vscode.ExtensionMode.Test) {
    void ctx.globalState.update(WELCOMED, true);
    void vscode.commands.executeCommand('workbench.action.openWalkthrough', 'mrpark10.deepfriend#deepfriend.start', false);
  }
}
