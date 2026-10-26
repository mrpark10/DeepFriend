import type { ModelInfo } from '../server/types';

/** The model whose class contains the cursor; otherwise the first model with blocks. */
export function pickModel(models: ModelInfo[], cursorLine: number): ModelInfo | undefined {
  const inside = models.find((m) => m.line <= cursorLine && cursorLine <= m.endLine);
  if (inside) {
    return inside;
  }
  return models.find((m) => Array.isArray(m.blocks) && m.blocks.length > 0) ?? models[0];
}

export function formatCount(n: number | null | undefined): string {
  if (n === null || n === undefined) {
    return '?';
  }
  const units: [number, string][] = [
    [1e9, 'B'],
    [1e6, 'M'],
    [1e3, 'K'],
  ];
  for (const [v, s] of units) {
    if (n >= v) {
      return (n / v).toFixed(1) + s;
    }
  }
  return String(n);
}

/** Decimal units (1 MB = 1,000,000 bytes), the way most learners read sizes. */
export function formatBytes(b: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = b;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i++;
  }
  return i === 0 ? `${Math.round(v)} B` : `${v.toFixed(1)} ${units[i]}`;
}
