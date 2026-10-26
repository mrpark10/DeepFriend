// Data & label samples: draws each stored sample on a canvas with its label.
(function () {
  const data = JSON.parse(document.getElementById('samples-data').textContent);
  const grid = document.getElementById('grid');
  for (const item of data.items) {
    const cell = document.createElement('div');
    cell.className = 'sample';
    const canvas = document.createElement('canvas');
    const scale = 4;
    const [h, w, c] = item.shape.length === 3 ? item.shape : [1, item.shape[0], 1];
    canvas.width = w * scale;
    canvas.height = Math.max(1, h) * scale * (item.shape.length === 3 ? 1 : 6);
    const ctx = canvas.getContext('2d');
    const rows = item.shape.length === 3 ? h : 1;
    const cellH = canvas.height / rows;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < w; x++) {
        let r, g, b;
        const i = (y * w + x) * c;
        if (c === 3) { r = item.pixels[i]; g = item.pixels[i + 1]; b = item.pixels[i + 2]; }
        else { r = g = b = item.pixels[i]; }
        ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
        ctx.fillRect(x * scale, y * cellH, scale, cellH);
      }
    }
    const label = document.createElement('div');
    label.className = 'muted';
    label.textContent = data.labelText.replace('{label}', item.label === null || item.label === undefined ? '?' : String(item.label));
    cell.appendChild(canvas);
    cell.appendChild(label);
    grid.appendChild(cell);
  }
})();
