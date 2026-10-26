import * as crypto from 'node:crypto';
import * as vscode from 'vscode';

export function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

export interface PageOptions {
  title: string;
  body: string;
  scripts?: string[]; // paths relative to the extension root, e.g. media/webview/card.js
  styles?: string[];
  lang?: string;
}

/** A webview page with a strict Content-Security-Policy (no inline scripts or remote content). */
export function webviewHtml(webview: vscode.Webview, extensionUri: vscode.Uri, opts: PageOptions): string {
  const nonce = crypto.randomBytes(16).toString('base64');
  const uri = (p: string) => webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, ...p.split('/'))).toString();
  const styles = ['media/webview/common.css', ...(opts.styles ?? [])].map((s) => `<link rel="stylesheet" href="${uri(s)}">`).join('\n');
  const scripts = (opts.scripts ?? []).map((s) => `<script nonce="${nonce}" src="${uri(s)}"></script>`).join('\n');
  const csp = [
    "default-src 'none'",
    `img-src ${webview.cspSource} data:`,
    `style-src ${webview.cspSource}`,
    `font-src ${webview.cspSource}`,
    `script-src 'nonce-${nonce}'`,
  ].join('; ');
  return `<!DOCTYPE html>
<html lang="${opts.lang ?? 'ko'}">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${escapeHtml(opts.title)}</title>
${styles}
</head>
<body>
${opts.body}
${scripts}
</body>
</html>`;
}
