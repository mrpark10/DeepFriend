import * as vscode from 'vscode';
import { config } from '../config';
import { t } from '../i18n/i18n';
import type { AnalysisResult, Diag, ModelInfo } from '../server/types';
import { AnalysisStore } from './analysisStore';
import { diagMessage, fixTitle, hintTooltipLines, normalizeInputShape } from './messages';

const SOURCE = 'DeepFriend';

function range(d: { line: number; col: number; endLine: number; endCol: number }): vscode.Range {
  return new vscode.Range(d.line, d.col, d.endLine, d.endCol);
}

function findDoc(uri: string): vscode.TextDocument | undefined {
  return vscode.workspace.textDocuments.find((d) => d.uri.toString() === uri);
}

/** Wires F1 (shape hints, red squiggles, quick fixes, "set input shape" CodeLens). */
export function registerF1(ctx: vscode.ExtensionContext, store: AnalysisStore): { refresh(): void } {
  const diagnostics = vscode.languages.createDiagnosticCollection('deepfriend');
  const hintsChanged = new vscode.EventEmitter<void>();
  const lensChanged = new vscode.EventEmitter<void>();
  const diagSource = new Map<string, Diag[]>();
  ctx.subscriptions.push(diagnostics, hintsChanged, lensChanged);

  const renderDiagnostics = (uri: string): void => {
    const res = store.get(uri);
    const list: vscode.Diagnostic[] = [];
    const raw: Diag[] = [];
    for (const m of res?.models ?? []) {
      for (const d of m.diagnostics) {
        const diag = new vscode.Diagnostic(range(d), diagMessage(d), vscode.DiagnosticSeverity.Error);
        diag.source = SOURCE;
        diag.code = d.code;
        list.push(diag);
        raw.push(d);
      }
    }
    diagSource.set(uri, raw);
    diagnostics.set(vscode.Uri.parse(uri), list);
  };

  const refresh = (): void => {
    for (const doc of vscode.workspace.textDocuments) {
      renderDiagnostics(doc.uri.toString());
    }
    hintsChanged.fire();
    lensChanged.fire();
  };

  ctx.subscriptions.push(
    store.onDidChange((uri) => {
      renderDiagnostics(uri);
      hintsChanged.fire();
      lensChanged.fire();
    }),
    vscode.workspace.onDidCloseTextDocument((doc) => {
      diagnostics.delete(doc.uri);
      diagSource.delete(doc.uri.toString());
    }),
  );

  const selector: vscode.DocumentSelector = [{ language: 'python' }];

  ctx.subscriptions.push(
    vscode.languages.registerInlayHintsProvider(selector, {
      onDidChangeInlayHints: hintsChanged.event,
      provideInlayHints(doc, visible) {
        if (!config.shapeHints()) {
          return [];
        }
        const res = store.get(doc.uri.toString());
        if (!res) {
          return [];
        }
        const out: vscode.InlayHint[] = [];
        for (const m of res.models) {
          for (const h of m.hints) {
            if (h.line < visible.start.line || h.line > visible.end.line || h.line >= doc.lineCount) {
              continue;
            }
            const pos = doc.lineAt(h.line).range.end;
            const hint = new vscode.InlayHint(pos, `→ ${h.shape}`);
            hint.paddingLeft = true;
            hint.tooltip = new vscode.MarkdownString(hintTooltipLines(h, m.verified).join('\n\n'));
            out.push(hint);
          }
        }
        return out;
      },
    }),
  );

  ctx.subscriptions.push(
    vscode.languages.registerCodeActionsProvider(
      selector,
      {
        provideCodeActions(doc, _range, context) {
          const raw = diagSource.get(doc.uri.toString()) ?? [];
          const actions: vscode.CodeAction[] = [];
          for (const vd of context.diagnostics) {
            if (vd.source !== SOURCE) {
              continue;
            }
            const d = raw.find((r) => r.code === vd.code && range(r).isEqual(vd.range));
            const title = d && fixTitle(d);
            if (!d?.fix || !title) {
              continue;
            }
            const action = new vscode.CodeAction(title, vscode.CodeActionKind.QuickFix);
            action.diagnostics = [vd];
            action.isPreferred = true;
            action.edit = new vscode.WorkspaceEdit();
            action.edit.replace(doc.uri, range(d.fix), d.fix.newText);
            actions.push(action);
          }
          return actions;
        },
      },
      { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix] },
    ),
  );

  ctx.subscriptions.push(
    vscode.languages.registerCodeLensProvider(selector, {
      onDidChangeCodeLenses: lensChanged.event,
      provideCodeLenses(doc) {
        const res = store.get(doc.uri.toString());
        return (res?.models ?? [])
          .filter((m) => m.needsInput)
          .map(
            (m) =>
              new vscode.CodeLens(new vscode.Range(m.line, 0, m.line, 0), {
                title: t('lens.setInput'),
                tooltip: t('lens.setInputTooltip'),
                command: 'deepfriend.setInputShape',
                arguments: [doc.uri.toString(), m.line, m.suggestedInput, m.kind],
              }),
          );
      },
    }),
  );

  ctx.subscriptions.push(
    vscode.commands.registerCommand(
      'deepfriend.setInputShape',
      async (uri?: string, line?: number, suggested?: string | null, kind?: ModelInfo['kind'], preset?: string) => {
        const doc = uri ? findDoc(uri) ?? (await vscode.workspace.openTextDocument(vscode.Uri.parse(uri))) : vscode.window.activeTextEditor?.document;
        if (!doc || line === undefined) {
          return;
        }
        const value =
          preset ??
          (await vscode.window.showInputBox({
            prompt: t(kind === 'keras' ? 'input.kerasPrompt' : 'input.prompt'),
            placeHolder: t('input.placeholder'),
            value: suggested ?? (kind === 'keras' ? '(None, 28, 28, 1)' : '(B, 3, 32, 32)'),
            validateInput: (v) => (normalizeInputShape(v) ? null : t('input.invalid')),
          }));
        const shape = value && normalizeInputShape(value);
        if (!shape) {
          return;
        }
        let insertAt = line;
        while (insertAt > 0 && doc.lineAt(insertAt - 1).text.trim().startsWith('@')) {
          insertAt--;
        }
        const indent = doc.lineAt(line).text.match(/^\s*/)?.[0] ?? '';
        const edit = new vscode.WorkspaceEdit();
        edit.insert(doc.uri, new vscode.Position(insertAt, 0), `${indent}# df: input ${shape}\n`);
        await vscode.workspace.applyEdit(edit);
      },
    ),
  );

  return { refresh };
}

export function resultFor(store: AnalysisStore, doc: vscode.TextDocument): AnalysisResult | undefined {
  return store.get(doc.uri.toString());
}
