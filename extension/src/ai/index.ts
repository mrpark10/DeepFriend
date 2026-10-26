import * as vscode from 'vscode';
import { config } from '../config';
import { getLanguage, t } from '../i18n/i18n';
import { AiCore } from './core';
import { currentProvider, SECRET_ANTHROPIC, SECRET_OPENAI, watchLmModels } from './providers';

/** Consent rules: never → no; always → yes; ask → modal only for explicit button presses. */
async function consent(preview: string, interactive: boolean): Promise<boolean> {
  const mode = config.aiConsent();
  if (mode === 'never') {
    return false;
  }
  if (mode === 'always') {
    return true;
  }
  if (!interactive) {
    return false;
  }
  const once = t('ai.consent.once');
  const always = t('ai.consent.always');
  const never = t('ai.consent.never');
  const shown = preview.length > 1500 ? preview.slice(0, 1500) + '\n…' : preview;
  const pick = await vscode.window.showInformationMessage(t('ai.consent.title'), { modal: true, detail: shown }, once, always, never);
  if (pick === always) {
    await config.set('ai.consent', 'always');
    return true;
  }
  if (pick === never) {
    await config.set('ai.consent', 'never');
    return false;
  }
  return pick === once;
}

export function registerAi(ctx: vscode.ExtensionContext): AiCore {
  watchLmModels(ctx.subscriptions);
  const core = new AiCore({
    provider: () => currentProvider(ctx.secrets),
    consent,
    language: () => getLanguage(),
  });
  ctx.subscriptions.push(
    vscode.commands.registerCommand('deepfriend.setApiKey', async () => {
      const which = await vscode.window.showQuickPick(
        [
          { label: 'Anthropic (Claude)', id: SECRET_ANTHROPIC, provider: 'anthropic' },
          { label: 'OpenAI-compatible', id: SECRET_OPENAI, provider: 'openai-compatible' },
        ],
        { title: t('ai.key.pick') },
      );
      if (!which) {
        return;
      }
      const key = await vscode.window.showInputBox({ title: which.label, prompt: t('ai.key.prompt'), password: true, ignoreFocusOut: true });
      if (key === undefined) {
        return;
      }
      if (key.trim()) {
        await ctx.secrets.store(which.id, key.trim());
        await config.set('ai.provider', which.provider);
        void vscode.window.showInformationMessage(t('ai.key.saved'));
      } else {
        await ctx.secrets.delete(which.id);
      }
    }),
  );
  return core;
}
