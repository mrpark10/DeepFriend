import * as vscode from 'vscode';
import { config } from '../config';
import type { Provider } from './core';

export const SECRET_ANTHROPIC = 'deepfriend.anthropicKey';
export const SECRET_OPENAI = 'deepfriend.openaiKey';

let lmModelCount = 0;

/** Track whether VS Code currently offers any chat model, so the AI button only shows when it can work. */
export function watchLmModels(subs: vscode.Disposable[]): void {
  const lm = (vscode as unknown as { lm?: typeof vscode.lm }).lm;
  if (!lm?.selectChatModels) {
    return;
  }
  const refresh = () =>
    void lm.selectChatModels({}).then(
      (m) => (lmModelCount = m.length),
      () => (lmModelCount = 0),
    );
  refresh();
  if (lm.onDidChangeChatModels) {
    subs.push(lm.onDidChangeChatModels(refresh));
  }
}

/** VS Code's built-in language models (e.g. GitHub Copilot) — no API key needed. */
function vscodeProvider(): Provider | null {
  const lm = (vscode as unknown as { lm?: typeof vscode.lm }).lm;
  if (!lm?.selectChatModels || lmModelCount === 0) {
    return null;
  }
  return {
    async complete(system, user, signal) {
      const models = await lm.selectChatModels({});
      const model = models[0];
      if (!model) {
        return null;
      }
      const cts = new vscode.CancellationTokenSource();
      signal.addEventListener('abort', () => cts.cancel());
      const res = await model.sendRequest([vscode.LanguageModelChatMessage.User(system + '\n\n' + user)], {}, cts.token);
      let text = '';
      for await (const part of res.text) {
        text += part;
      }
      return text;
    },
  };
}

/** Anthropic Claude via the official SDK, with the user's own API key. */
function anthropicProvider(secrets: vscode.SecretStorage): Provider {
  return {
    async complete(system, user, signal) {
      const apiKey = await secrets.get(SECRET_ANTHROPIC);
      if (!apiKey) {
        return null;
      }
      const { default: Anthropic } = await import('@anthropic-ai/sdk');
      const client = new Anthropic({ apiKey, maxRetries: 1 });
      try {
        const response = await client.beta.messages.create(
          {
            model: config.anthropicModel() || 'claude-opus-5-5',
            max_tokens: 4000,
            // Short rewording task: keep reasoning light. Fallback keeps answers coming if a request is declined.
            output_config: { effort: 'low' },
            betas: ['server-side-fallback-2026-07-01'],
            fallbacks: 'default',
            system,
            messages: [{ role: 'user', content: user }],
          },
          { signal },
        );
        if (response.stop_reason === 'refusal' || response.stop_reason === 'max_tokens') {
          return null;
        }
        return response.content.map((b) => (b.type === 'text' ? b.text : '')).join('');
      } catch (e) {
        if (e instanceof Anthropic.AuthenticationError) {
          void vscode.window.showWarningMessage('DeepFriend: Anthropic API key was rejected.');
        }
        return null;
      }
    },
  };
}

/** Any OpenAI-compatible chat endpoint the user chose (OpenAI, Gemini, a local Ollama, ...). */
function openAiCompatibleProvider(secrets: vscode.SecretStorage): Provider {
  return {
    async complete(system, user, signal) {
      const key = (await secrets.get(SECRET_OPENAI)) ?? '';
      const base = (config.openaiBaseUrl() || 'https://api.openai.com/v1').replace(/\/+$/, '');
      const res = await fetch(base + '/chat/completions', {
        method: 'POST',
        signal,
        headers: { 'content-type': 'application/json', ...(key ? { authorization: 'Bearer ' + key } : {}) },
        body: JSON.stringify({
          model: config.openaiModel() || 'gpt-4o-mini',
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: user },
          ],
        }),
      });
      if (!res.ok) {
        return null;
      }
      const j = (await res.json()) as { choices?: { message?: { content?: string } }[] };
      return j.choices?.[0]?.message?.content ?? null;
    },
  };
}

export function currentProvider(secrets: vscode.SecretStorage): Provider | null {
  switch (config.aiProvider()) {
    case 'vscode':
      return vscodeProvider();
    case 'anthropic':
      return anthropicProvider(secrets);
    case 'openai-compatible':
      return openAiCompatibleProvider(secrets);
    default:
      return null;
  }
}
