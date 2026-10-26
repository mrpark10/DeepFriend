import * as vscode from 'vscode';
import { config } from '../config';
import type { AnalysisStore } from '../f1/analysisStore';
import { CONCEPTS } from './concepts';
import { hoverMarkdown } from './hoverText';
import { resolveAt } from './resolve';

/** F7: plain-language explanations when hovering layer names and their parameters. */
export function registerF7(ctx: vscode.ExtensionContext, store: AnalysisStore): void {
  ctx.subscriptions.push(
    vscode.languages.registerHoverProvider({ language: 'python' }, {
      provideHover(doc, pos) {
        const text = doc.getText();
        const r = resolveAt(text, doc.offsetAt(pos));
        if (!r) {
          return undefined;
        }
        const res = store.get(doc.uri.toString());
        const hint = res?.models.flatMap((m) => m.hints).find((h) => h.line === pos.line && h.shape !== '?');
        const md = new vscode.MarkdownString(hoverMarkdown(CONCEPTS[r.key], config.level(), r.param, hint?.shape));
        md.isTrusted = false;
        return new vscode.Hover(md, doc.getWordRangeAtPosition(pos));
      },
    }),
  );
}
