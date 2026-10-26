// Compare two runs: overlaid training and validation loss curves.
(function () {
  const vscode = acquireVsCodeApi();
  const data = JSON.parse(document.getElementById('compare-data').textContent);
  const box = document.getElementById('charts');
  const css = (n, f) => getComputedStyle(document.body).getPropertyValue(n).trim() || f;
  const colA = css('--vscode-charts-blue', '#3794ff');
  const colB = css('--vscode-charts-orange', '#d18616');
  const fg = css('--vscode-foreground', '#ccc');
  const grid = css('--vscode-editorWidget-border', '#8884');

  function merge(xa, ya, xb, yb) {
    const xs = Array.from(new Set(xa.concat(xb))).sort((p, q) => p - q);
    const ia = new Map(xa.map((x, i) => [x, ya[i]]));
    const ib = new Map(xb.map((x, i) => [x, yb[i]]));
    return [xs, xs.map((x) => (ia.has(x) ? ia.get(x) : null)), xs.map((x) => (ib.has(x) ? ib.get(x) : null))];
  }

  function plot(title, xLabel, rows) {
    const wrap = document.createElement('div');
    const h = document.createElement('p');
    h.className = 'chart-title';
    h.textContent = title;
    wrap.appendChild(h);
    box.appendChild(wrap);
    if (!rows[0].length) return;
    const opts = {
      width: Math.max(300, wrap.clientWidth || 360),
      height: 200,
      scales: { x: { time: false } },
      axes: [{ stroke: fg, grid: { stroke: grid }, label: xLabel }, { stroke: fg, grid: { stroke: grid } }],
      series: [{ label: xLabel }, { label: 'A · ' + data.a.name, stroke: colA, width: 1.5, spanGaps: true }, { label: 'B · ' + data.b.name, stroke: colB, width: 1.5, spanGaps: true }],
    };
    new uPlot(opts, rows, wrap);
  }

  plot(data.L.trainLoss, data.L.step, merge(data.a.chart.x, data.a.chart.trainLoss, data.b.chart.x, data.b.chart.trainLoss));
  const ea = data.a.epochs.filter((e) => e.val_loss !== undefined);
  const eb = data.b.epochs.filter((e) => e.val_loss !== undefined);
  plot(data.L.valLoss, data.L.epoch, merge(ea.map((e) => e.epoch + 1), ea.map((e) => e.val_loss), eb.map((e) => e.epoch + 1), eb.map((e) => e.val_loss)));

  document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-file]');
    if (b) vscode.postMessage({ type: 'file', file: Number(b.getAttribute('data-file')) });
  });
})();
