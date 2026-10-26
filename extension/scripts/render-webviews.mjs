// Screenshot QA: renders each webview with fixture data in headless Chrome (dark/light, narrow/wide).
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const ext = join(here, '..');
const outDir = join(ext, '..', 'qa', 'screens');
const tmp = join(ext, 'out', 'qa-pages');
mkdirSync(outDir, { recursive: true });
mkdirSync(tmp, { recursive: true });
const fx = JSON.parse(readFileSync(join(ext, 'out', 'webview-fixtures.json'), 'utf8'));
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const THEMES = {
  dark: { '--vscode-foreground': '#cccccc', '--vscode-editor-background': '#1e1e1e', '--vscode-sideBar-background': '#252526', '--vscode-descriptionForeground': '#9d9d9d', '--vscode-textLink-foreground': '#3794ff', '--vscode-button-background': '#0e639c', '--vscode-button-foreground': '#ffffff', '--vscode-button-hoverBackground': '#1177bb', '--vscode-button-secondaryBackground': '#3a3d41', '--vscode-button-secondaryForeground': '#ffffff', '--vscode-badge-background': '#4d4d4d', '--vscode-badge-foreground': '#ffffff', '--vscode-panel-border': '#80808059', '--vscode-widget-border': '#ffffff1f', '--vscode-textCodeBlock-background': '#0a0a0a66', '--vscode-editorWidget-background': '#252526', '--vscode-focusBorder': '#007fd4', '--vscode-editorError-foreground': '#f14c4c', '--vscode-editorWarning-foreground': '#cca700', '--vscode-testing-iconPassed': '#73c991', '--vscode-charts-blue': '#3794ff', '--vscode-charts-orange': '#d18616', '--vscode-editorWidget-border': '#454545', '--vscode-diffEditor-insertedLineBackground': '#9bb95533' },
  light: { '--vscode-foreground': '#3b3b3b', '--vscode-editor-background': '#ffffff', '--vscode-sideBar-background': '#f8f8f8', '--vscode-descriptionForeground': '#717171', '--vscode-textLink-foreground': '#005fb8', '--vscode-button-background': '#005fb8', '--vscode-button-foreground': '#ffffff', '--vscode-button-hoverBackground': '#0258a8', '--vscode-button-secondaryBackground': '#e5e5e5', '--vscode-button-secondaryForeground': '#3b3b3b', '--vscode-badge-background': '#cccccc', '--vscode-badge-foreground': '#3b3b3b', '--vscode-panel-border': '#e5e5e5', '--vscode-widget-border': '#e5e5e5', '--vscode-textCodeBlock-background': '#f2f2f2', '--vscode-editorWidget-background': '#f8f8f8', '--vscode-focusBorder': '#005fb8', '--vscode-editorError-foreground': '#e51400', '--vscode-editorWarning-foreground': '#bf8803', '--vscode-testing-iconPassed': '#388a34', '--vscode-charts-blue': '#1a85ff', '--vscode-charts-orange': '#d18616', '--vscode-editorWidget-border': '#c8c8c8', '--vscode-diffEditor-insertedLineBackground': '#9bb95533' },
};

function page(name, theme, body, scripts, styles, message) {
  const vars = Object.entries(THEMES[theme]).map(([k, v]) => `${k}:${v}`).join(';');
  const css = ['media/webview/common.css', ...styles].map((s) => `<link rel="stylesheet" href="file://${join(ext, s)}">`).join('');
  const js = scripts.map((s) => `<script src="file://${join(ext, s)}"></script>`).join('');
  const html = `<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">${css}<style>:root{${vars};--vscode-font-family:-apple-system,"Apple SD Gothic Neo",sans-serif;--vscode-font-size:13px;--vscode-editor-font-family:Menlo,monospace;--vscode-editor-font-size:12px}</style>
<script>window.acquireVsCodeApi=()=>({postMessage:()=>{}});</script></head><body>${body}${js}
<script>${message ? `setTimeout(()=>window.postMessage(${JSON.stringify(message)},'*'),50);` : ''}</script></body></html>`;
  const file = join(tmp, `${name}.html`);
  writeFileSync(file, html);
  return file;
}

// Headless Chrome enforces a minimum window width of 500px, so narrow views are rendered
// inside an iframe of the real width (otherwise the screenshot just crops a 500px layout).
function shot(file, png, width, height) {
  let target = file;
  const win = Math.max(width, 500);
  if (width < 500) {
    target = file.replace(/\.html$/, `.frame${width}.html`);
    writeFileSync(target, `<!DOCTYPE html><html><body style="margin:0;background:#777"><iframe src="file://${file}" style="border:0;width:${width}px;height:${height}px;display:block"></iframe></body></html>`);
  }
  execFileSync(CHROME, ['--headless=new', '--disable-gpu', '--hide-scrollbars', '--allow-file-access-from-files', `--window-size=${win},${height}`, '--virtual-time-budget=1500', `--screenshot=${png}`, `file://${target}`], { stdio: 'ignore' });
}

const jobs = [];
for (const theme of ['dark', 'light']) {
  for (const [state, w, h] of [['overfit', 900, 620], ['nan', 380, 900], ['healthy', 900, 520], ['chance', 380, 900], ['empty', 600, 260]]) {
    const lang = state === 'healthy' ? 'en' : 'ko';
    const f = page(`doctor-${state}-${theme}`, theme, '<div id="root"></div>', ['media/vendor/uPlot.iife.min.js', 'media/webview/doctor.js'], ['media/vendor/uPlot.min.css', 'media/webview/doctor.css'], fx[`doctor-${state}-${lang}`]);
    jobs.push([f, join(outDir, `doctor-${state}-${theme}-${w}.png`), w, h]);
  }
  const model = { type: 'model', L: { verified: 'PyTorch로 확인함', noBlocks: '레이어를 찾지 못했어요.' }, totalParams: 61706, activeLine: 22, empty: '', scroll: false,
    model: { name: 'LeNet', verified: true, paramsText: '파라미터 61.7K개', memoryText: '학습할 때 대략 1.0 MB 필요', memoryDetail: '', input: '입력 (1, 1, 32, 32)',
      blocks: [['conv1', 'Conv2d', '(1, 6, 28, 28)', '156개', 20], ['pool', 'MaxPool2d', '(1, 6, 14, 14)', '0개', 21], ['conv2', 'Conv2d', '(1, 16, 10, 10)', '2.4K개', 22], ['pool', 'MaxPool2d', '(1, 16, 5, 5)', '0개', 21], ['fc1', 'Linear', '(1, 120)', '48.1K개', 23], ['fc2', 'Linear', '(1, 84)', '10.2K개', 24], ['fc3', 'Linear', '?', '850개', 25]]
        .map(([name, kind, outShape, paramsText, line]) => ({ name, kind, outShape, paramsText, line, error: name === 'fc2', unknown: outShape === '?' })) } };
  jobs.push([page(`model-${theme}`, theme, '<div id="root"></div>', ['media/webview/model.js'], ['media/webview/model.css'], model), join(outDir, `model-${theme}-300.png`), 300, 760]);
  const card = `<div class="card"><h1>크기 불일치: Linear 층에 들어가는 숫자 개수가 달라요</h1><p class="muted">train.py 6번째 줄</p><h2>무슨 뜻이에요</h2><p>앞 레이어가 12544개의 숫자를 내보내는데, 다음 Linear 층은 10000개를 받도록 만들어져 있어요.</p><h2>왜 그래요</h2><p>보통 Flatten(또는 view) 다음 첫 Linear의 in_features를 잘못 계산했거나, 입력 이미지 크기가 생각과 달라서 생겨요.</p><h2>이렇게 해보세요</h2><p>그 Linear 층의 in_features를 12544로 바꿔 보세요.</p></div><div class="actions"><button>에러 난 줄로 이동</button><button class="secondary">검색해 보기</button></div><h2>원래 에러 메시지</h2><pre>RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x12544 and 10000x10)</pre>`;
  jobs.push([page(`card-${theme}`, theme, card, [], [], null), join(outDir, `card-${theme}-520.png`), 520, 640]);
  const samples = `<h1>데이터·라벨 샘플</h1><div id="grid" class="samples"></div><script type="application/json" id="samples-data">${JSON.stringify({ ...fx['samples-ko'], labelText: '라벨: {label}' })}</script>`;
  jobs.push([page(`samples-${theme}`, theme, samples, ['media/webview/samples.js'], ['media/webview/samples.css'], null), join(outDir, `samples-${theme}-700.png`), 700, 420]);
}
for (const [f, png, w, h] of jobs) {
  shot(f, png, w, h);
  console.log(png);
}
