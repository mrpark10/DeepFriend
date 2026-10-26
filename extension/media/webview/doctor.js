// Training doctor panel: charts (uPlot) + diagnosis cards. Data arrives via postMessage.
(function () {
  const vscode = acquireVsCodeApi();
  const root = document.getElementById('root');
  let plots = [];
  let last = null;

  function el(tag, attrs, children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === 'class') e.className = v;
      else if (k === 'text') e.textContent = v;
      else e.setAttribute(k, v);
    }
    for (const c of children || []) if (c) e.appendChild(c);
    return e;
  }

  function cssVar(name, fallback) {
    const v = getComputedStyle(document.body).getPropertyValue(name).trim();
    return v || fallback;
  }

  function makePlot(container, title, x, series, labels) {
    const width = Math.max(260, container.clientWidth || 400);
    const fg = cssVar('--vscode-foreground', '#ccc');
    const grid = cssVar('--vscode-editorWidget-border', '#8884');
    const colors = [cssVar('--vscode-charts-blue', '#3794ff'), cssVar('--vscode-charts-orange', '#d18616')];
    const opts = {
      width,
      height: 170,
      legend: { show: true },
      cursor: { drag: { x: false, y: false } },
      scales: { x: { time: false } },
      axes: [
        { stroke: fg, grid: { stroke: grid, width: 1 }, label: labels.step },
        { stroke: fg, grid: { stroke: grid, width: 1 } },
      ],
      series: [{ label: labels.step }].concat(
        series.map((s, i) => ({ label: s.label, stroke: colors[i % colors.length], width: i === 0 ? 1.5 : 2, spanGaps: true, points: { show: i > 0, size: 5 } })),
      ),
    };
    const box = el('div', { class: 'chart' }, [el('p', { class: 'chart-title', text: title })]);
    container.appendChild(box);
    const plot = new uPlot(opts, [x].concat(series.map((s) => s.data)), box);
    plots.push({ plot, box });
  }

  function render(msg) {
    last = msg;
    plots.forEach((p) => p.plot.destroy());
    plots = [];
    root.innerHTML = '';
    const L = msg.L;
    if (!msg.run) {
      const code = el('code', { text: 'import deepfriend; deepfriend.watch()' });
      const btn = el('button', { class: 'secondary', text: L.copy });
      btn.addEventListener('click', () => {
        vscode.postMessage({ type: 'copy' });
        btn.textContent = L.copied;
      });
      root.appendChild(el('div', { class: 'empty card' }, [el('h1', { text: L.emptyTitle }), el('p', { text: L.emptyBody }), el('div', { class: 'oneliner' }, [code, btn])]));
      return;
    }
    const head = el('div', { class: 'head' }, [el('h1', { text: msg.run.title }), el('span', { class: 'badge', text: msg.run.status }), el('span', { class: 'muted', text: msg.run.meta })]);
    root.appendChild(head);
    const left = el('div', {});
    const right = el('div', {});
    root.appendChild(el('div', { class: 'layout' }, [left, right]));
    const c = msg.chart;
    if (c.x.length) {
      makePlot(left, L.loss, c.x, [{ label: L.train, data: c.trainLoss }, { label: L.val, data: c.valLoss }], L);
      if (c.hasAcc) makePlot(left, L.acc, c.x, [{ label: L.train, data: c.trainAcc }, { label: L.val, data: c.valAcc }], L);
    }
    if (!msg.findings.length) {
      right.appendChild(el('div', { class: 'card good' }, [el('p', { text: msg.note })]));
    }
    msg.findings.forEach((f, fi) => {
      const title = el('h2', { text: L.what });
      if (f.ai) title.appendChild(el('span', { class: 'badge', text: L.ai }));
      const actions = el('div', { class: 'actions' });
      f.actions.forEach((a, ai) => {
        const b = el('button', { class: ai ? 'secondary' : '', text: a });
        b.addEventListener('click', () => vscode.postMessage({ type: 'action', finding: fi, action: ai }));
        actions.appendChild(b);
      });
      right.appendChild(
        el('div', { class: 'card finding ' + f.id }, [title, el('p', { text: f.what }), el('h2', { text: L.why }), el('p', { text: f.why }), el('h2', { text: L.try }), el('p', { text: f.try }), actions]),
      );
    });
  }

  window.addEventListener('message', (e) => {
    if (e.data && e.data.type === 'state') render(e.data);
  });
  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => last && render(last), 150);
  });
  vscode.postMessage({ type: 'ready' });
})();
