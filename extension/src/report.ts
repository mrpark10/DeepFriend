import * as os from 'node:os';
import * as vscode from 'vscode';
import { t } from './i18n/i18n';
import { buildIssueUrl } from './reportUrl';
import type { InitializeResult } from './server/types';

export async function openIssue(ctx: vscode.ExtensionContext, info: InitializeResult | null, error?: unknown): Promise<void> {
  const err = error instanceof Error ? `${error.name}: ${error.message}\n${error.stack ?? ''}` : error ? String(error) : undefined;
  const url = buildIssueUrl({
    title: t('report.title'),
    intro: t('report.bodyIntro'),
    extVersion: String(ctx.extension.packageJSON.version ?? ''),
    vscodeVersion: vscode.version,
    os: `${os.platform()} ${os.arch()} ${os.release()}`,
    python: info?.pythonVersion ?? null,
    torch: info?.torch ?? null,
    tensorflow: info?.tensorflow ?? null,
    error: err,
  });
  await vscode.env.openExternal(vscode.Uri.parse(url));
}
