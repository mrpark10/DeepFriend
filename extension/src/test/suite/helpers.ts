import * as path from 'path';
import * as vscode from 'vscode';
import type { DeepFriendApi } from '../../extension';

export async function api(): Promise<DeepFriendApi> {
  const ext = vscode.extensions.getExtension<DeepFriendApi>('mrpark10.deepfriend');
  if (!ext) {
    throw new Error('extension not found');
  }
  return ext.activate();
}

export function wsPath(...parts: string[]): string {
  return path.join(vscode.workspace.workspaceFolders![0].uri.fsPath, ...parts);
}

export async function waitFor<T>(fn: () => T | undefined | null | false | Promise<T | undefined | null | false>, timeoutMs = 20000, stepMs = 100): Promise<T> {
  const start = Date.now();
  for (;;) {
    const v = await fn();
    if (v) {
      return v;
    }
    if (Date.now() - start > timeoutMs) {
      throw new Error('waitFor timed out');
    }
    await new Promise((r) => setTimeout(r, stepMs));
  }
}
