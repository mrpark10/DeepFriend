// Model structure view: vertical blocks, click to jump to the code line.
(function () {
  const vscode = acquireVsCodeApi();
  const root = document.getElementById('root');

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function render(msg) {
    root.innerHTML = '';
    if (!msg.model) {
      root.appendChild(el('p', 'muted', msg.empty));
      return;
    }
    const m = msg.model;
    const summary = el('div', 'summary');
    const title = el('div', 'big', m.name);
    if (m.verified) title.appendChild(el('span', 'badge', msg.L.verified));
    summary.appendChild(title);
    summary.appendChild(el('div', 'muted', m.paramsText));
    if (m.memoryText) {
      const mem = el('div', 'muted', m.memoryText);
      mem.title = m.memoryDetail;
      summary.appendChild(mem);
    }
    root.appendChild(summary);
    const flow = el('div', 'flow');
    if (m.input) flow.appendChild(el('div', 'input', m.input));
    m.blocks.forEach((b, i) => {
      if (i > 0 || m.input) flow.appendChild(el('div', 'arrow', '↓'));
      const box = el('div', 'block' + (b.error ? ' error' : '') + (b.unknown ? ' unknown' : '') + (b.line === msg.activeLine ? ' active' : ''));
      box.setAttribute('role', 'button');
      box.tabIndex = 0;
      const r1 = el('div', 'row');
      r1.appendChild(el('span', 'name', b.name));
      r1.appendChild(el('span', 'kind', b.kind));
      const r2 = el('div', 'row');
      r2.appendChild(el('span', 'shape', '→ ' + b.outShape));
      r2.appendChild(el('span', 'params', b.paramsText));
      box.appendChild(r1);
      box.appendChild(r2);
      const go = () => vscode.postMessage({ type: 'reveal', line: b.line });
      box.addEventListener('click', go);
      box.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') go();
      });
      flow.appendChild(box);
    });
    if (!m.blocks.length) flow.appendChild(el('p', 'muted', msg.L.noBlocks));
    root.appendChild(flow);
    const active = root.querySelector('.block.active');
    if (active && msg.scroll) active.scrollIntoView({ block: 'nearest' });
  }

  window.addEventListener('message', (e) => {
    if (e.data && e.data.type === 'model') render(e.data);
  });
  vscode.postMessage({ type: 'ready' });
})();
