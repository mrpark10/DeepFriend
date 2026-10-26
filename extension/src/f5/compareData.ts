import * as fs from 'node:fs';
import * as path from 'node:path';
import { formatNumber } from '../f2/doctorText';
import type { RunMeta } from '../f2/runStore';

export interface DiffRow {
  key: string;
  a: string;
  b: string;
  same: boolean;
}

function show(v: unknown): string {
  if (v === undefined || v === null) {
    return '-';
  }
  if (typeof v === 'number') {
    return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(4)));
  }
  return typeof v === 'string' ? v : JSON.stringify(v);
}

function flatten(m: RunMeta): Record<string, unknown> {
  const out: Record<string, unknown> = { ...(m.config ?? {}) };
  out['model.params'] = m.model?.params;
  out['numClasses'] = m.numClasses;
  for (const [k, v] of Object.entries(m.final ?? {})) {
    out['final.' + k] = v;
  }
  return out;
}

/** Settings table for comparing two runs: differing rows first, each group alphabetical. */
export function configDiff(a: RunMeta, b: RunMeta): DiffRow[] {
  const fa = flatten(a);
  const fb = flatten(b);
  const keys = [...new Set([...Object.keys(fa), ...Object.keys(fb)])].filter((k) => fa[k] !== undefined || fb[k] !== undefined);
  const rows = keys.map((key) => {
    const va = show(fa[key]);
    const vb = show(fb[key]);
    return { key, a: va, b: vb, same: va === vb };
  });
  const isFinal = (k: string) => (k.startsWith('final.') ? 1 : 0);
  return rows.sort((x, y) => Number(x.same) - Number(y.same) || isFinal(x.key) - isFinal(y.key) || x.key.localeCompare(y.key));
}

/** The one number shown next to a run in the list. */
export function headlineMetric(m: RunMeta): { key: string; text: string } | null {
  const f = m.final ?? {};
  for (const key of ['val_acc', 'acc']) {
    if (typeof f[key] === 'number') {
      return { key, text: (f[key] * 100).toFixed(1) + '%' };
    }
  }
  for (const key of ['val_loss', 'loss']) {
    if (typeof f[key] === 'number') {
      return { key, text: formatNumber(f[key]) };
    }
  }
  return null;
}

function walk(dir: string, base: string, out: Set<string>): void {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) {
      walk(p, base, out);
    } else {
      out.add(path.relative(base, p).split(path.sep).join('/'));
    }
  }
}

/** Relative paths of all code snapshot files in either run. */
export function codeFiles(dirA: string, dirB: string): string[] {
  const out = new Set<string>();
  walk(path.join(dirA, 'code'), path.join(dirA, 'code'), out);
  walk(path.join(dirB, 'code'), path.join(dirB, 'code'), out);
  return [...out].sort();
}
