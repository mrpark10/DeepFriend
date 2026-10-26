import * as vscode from 'vscode';
import type { AnalysisStore } from '../f1/analysisStore';
import { getLanguage, t } from '../i18n/i18n';
import type { ModelInfo } from '../server/types';
import { webviewHtml } from '../webview/html';
import { formatBytes, formatCount, pickModel } from './modelData';

export interface ModelMessage {
  type: 'model';
  L: Record<string, string>;
  model: null | {
    name: string;
    verified: boolean;
    paramsText: string;
    memoryText: string;
    memoryDetail: string;
    input: string | null;
    blocks: { name: string; kind: string; outShape: string; paramsText: string; line: number; error: boolean; unknown: boolean }[];
  };
  totalParams: number | null;
  activeLine: number;
  empty: string;
  scroll: boolean;
}

export class ModelView implements vscode.WebviewViewProvider {
  private view: vscode.WebviewView | undefined;
  private docUri: string | undefined;
  lastMessage: ModelMessage | undefined;

  constructor(private extensionUri: vscode.Uri, private store: AnalysisStore) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = { enableScripts: true, localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, 'media')] };
    view.webview.html = webviewHtml(view.webview, this.extensionUri, {
      title: t('model.title'),
      body: '<div id="root"></div>',
      scripts: ['media/webview/model.js'],
      styles: ['media/webview/model.css'],
      lang: getLanguage(),
    });
    view.webview.onDidReceiveMessage((m: { type: string; line?: number }) => this.handleMessage(m));
    view.onDidChangeVisibility(() => view.visible && this.update(false));
  }

  async handleMessage(m: { type: string; line?: number }): Promise<void> {
    if (m.type === 'ready') {
      this.update(true);
    } else if (m.type === 'reveal' && typeof m.line === 'number' && this.docUri) {
      const doc = vscode.workspace.textDocuments.find((d) => d.uri.toString() === this.docUri);
      if (!doc) {
        return;
      }
      const line = Math.min(m.line, doc.lineCount - 1);
      const range = doc.lineAt(line).range;
      await vscode.window.showTextDocument(doc, { selection: new vscode.Range(range.start, range.start), preserveFocus: false, viewColumn: vscode.ViewColumn.One });
    }
  }

  update(scroll: boolean): void {
    const editor = vscode.window.activeTextEditor;
    const L = { verified: t('model.verified'), noBlocks: t('model.noBlocks') };
    let msg: ModelMessage;
    const res = editor && editor.document.languageId === 'python' ? this.store.get(editor.document.uri.toString()) : undefined;
    const cursor = editor?.selection.active.line ?? 0;
    const model = res ? pickModel(res.models, cursor) : undefined;
    if (!editor || !res || !model) {
      msg = { type: 'model', L, model: null, totalParams: null, activeLine: -1, empty: t('model.empty'), scroll };
    } else {
      this.docUri = editor.document.uri.toString();
      msg = { type: 'model', L, model: this.describe(model), totalParams: model.totalParams, activeLine: cursor, empty: '', scroll };
    }
    this.lastMessage = msg;
    void this.view?.webview.postMessage(msg);
  }

  private describe(m: ModelInfo): NonNullable<ModelMessage['model']> {
    const mem = m.memory;
    return {
      name: m.name,
      verified: m.verified,
      paramsText: t('model.params', { n: formatCount(m.totalParams) }),
      memoryText: mem ? t('model.memory', { total: formatBytes(mem.total) }) : '',
      memoryDetail: mem
        ? t('model.memoryDetail', { params: formatBytes(mem.params), grads: formatBytes(mem.grads), optimizer: formatBytes(mem.optimizer), activations: formatBytes(mem.activations) })
        : '',
      input: m.input ? t('model.input', { shape: m.input.shapes.join(', ') }) : null,
      blocks: m.blocks.map((b) => ({
        name: b.name,
        kind: b.kind,
        outShape: b.outShape,
        paramsText: b.params === null ? '' : t('model.blockParams', { n: formatCount(b.params) }),
        line: b.line,
        error: b.error,
        unknown: b.outShape === '?',
      })),
    };
  }
}

/** F4: model structure blocks in the DeepFriend sidebar. */
export function registerF4(ctx: vscode.ExtensionContext, store: AnalysisStore): ModelView {
  const view = new ModelView(ctx.extensionUri, store);
  let timer: ReturnType<typeof setTimeout> | undefined;
  const soon = (scroll: boolean) => {
    clearTimeout(timer);
    timer = setTimeout(() => view.update(scroll), 80);
  };
  ctx.subscriptions.push(
    vscode.window.registerWebviewViewProvider('deepfriend.model', view),
    store.onDidChange(() => soon(false)),
    vscode.window.onDidChangeActiveTextEditor(() => soon(true)),
    vscode.window.onDidChangeTextEditorSelection(() => soon(true)),
    vscode.commands.registerCommand('deepfriend.openModelView', async () => {
      await vscode.commands.executeCommand('deepfriend.model.focus');
      view.update(true);
    }),
  );
  return view;
}
