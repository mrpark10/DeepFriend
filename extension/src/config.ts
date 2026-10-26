import * as vscode from 'vscode';
import type { Language } from './i18n/i18n';

export type Level = 'beginner' | 'normal';
export type AiProvider = 'off' | 'vscode' | 'anthropic' | 'openai-compatible';
export type AiConsent = 'ask' | 'always' | 'never';

function cfg(): vscode.WorkspaceConfiguration {
  return vscode.workspace.getConfiguration('deepfriend');
}

export const config = {
  language: (): Language => (cfg().get<string>('language') === 'en' ? 'en' : 'ko'),
  level: (): Level => (cfg().get<string>('level') === 'normal' ? 'normal' : 'beginner'),
  pythonPath: (): string => cfg().get<string>('pythonPath') ?? '',
  shapeHints: (): boolean => cfg().get<boolean>('shapeHints.enabled') ?? true,
  runtimeCheck: (): boolean => cfg().get<boolean>('runtimeCheck') ?? true,
  autoRecord: (): boolean => cfg().get<boolean>('autoRecord') ?? true,
  aiProvider: (): AiProvider => (cfg().get<AiProvider>('ai.provider') ?? 'vscode'),
  aiConsent: (): AiConsent => (cfg().get<AiConsent>('ai.consent') ?? 'ask'),
  anthropicModel: (): string => cfg().get<string>('ai.anthropicModel') ?? '',
  openaiBaseUrl: (): string => cfg().get<string>('ai.openaiBaseUrl') ?? '',
  openaiModel: (): string => cfg().get<string>('ai.openaiModel') ?? '',
  errorReportingPrompt: (): boolean => cfg().get<boolean>('errorReporting.prompt') ?? true,
  async set(key: string, value: unknown, target = vscode.ConfigurationTarget.Global): Promise<void> {
    await cfg().update(key, value, target);
  },
};

export const FRAMEWORK_IMPORT = /^\s*(?:import|from)\s+(torch|tensorflow|keras|tf_keras)\b/m;

export function detectFramework(text: string): 'pytorch' | 'tensorflow' | null {
  const m = FRAMEWORK_IMPORT.exec(text);
  if (!m) {
    return null;
  }
  return m[1] === 'torch' ? 'pytorch' : 'tensorflow';
}
