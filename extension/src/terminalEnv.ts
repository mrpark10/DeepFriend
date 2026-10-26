import * as path from 'node:path';
import * as vscode from 'vscode';
import { config } from './config';
import { t } from './i18n/i18n';

/** Make `import deepfriend` work in integrated terminals without pip install. */
export function applyTerminalEnv(ctx: vscode.ExtensionContext, bundledPython: string): void {
  const env = ctx.environmentVariableCollection;
  env.description = t('terminal.envDescription');
  env.prepend('PYTHONPATH', bundledPython + path.delimiter);
  const folder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (folder) {
    env.replace('DEEPFRIEND_WORKSPACE', folder);
  }
  env.replace('DEEPFRIEND_RECORD', config.autoRecord() ? '1' : '0');
}
