import { StringKey, t } from '../i18n/i18n';
import { ko } from '../i18n/ko';
import type { Diag, Hint } from '../server/types';

function has(key: string): key is StringKey {
  return key in ko;
}

/** `t` with every placeholder filled, so users never see `{braces}`. */
function fill(key: StringKey, args: Record<string, string | number>): string {
  return t(key, args).replace(/\{[a-zA-Z0-9_]+\}/g, '?');
}

export function diagMessage(d: Diag): string {
  const args: Record<string, string | number> = { ...d.args };
  if (d.code === 'invalid_arg') {
    const rk = 'reason.' + String(args.reason ?? 'other');
    args.reason = t(has(rk) ? rk : 'reason.other');
    if (args.layer === undefined) {
      return args.reason as string;
    }
  }
  const key = 'diag.' + d.code;
  return has(key) ? fill(key, args) : String(d.code);
}

export function fixTitle(d: Diag): string | null {
  const key = 'fix.' + d.code;
  return has(key) ? fill(key, d.args) : null;
}

export function unknownText(reason: string | undefined): string {
  const r = reason ?? 'unknown';
  if (r.startsWith('unsupported_op:')) {
    return t('hint.unknown.unsupported', { op: r.slice('unsupported_op:'.length) });
  }
  const key = 'hint.unknown.' + r;
  return t(has(key) ? key : 'hint.unknown.unknown');
}

export function hintTooltipLines(h: Hint, verified: boolean): string[] {
  const lines: string[] = [];
  if (h.shape === '?') {
    lines.push(unknownText(h.unknownReason));
    return lines;
  }
  if (h.all && h.all.length > 1) {
    lines.push(t('hint.many'), ...h.all.map((s) => '- ' + s));
  }
  lines.push(t(verified ? 'hint.verified' : 'hint.static'));
  return lines;
}

const DIM = /^(?:[1-9][0-9]*|[A-Za-z_][A-Za-z0-9_]*)$/;

/** Validate a shape like "(32, 3, 224, 224)" or "(3, 5), (3, 7)"; returns the canonical form or null. */
export function normalizeInputShape(text: string): string | null {
  let s = text.trim();
  if (!s) {
    return null;
  }
  if (!s.startsWith('(')) {
    s = '(' + s + ')';
  }
  const tuples: string[] = [];
  const re = /\(([^()]*)\)\s*(,\s*|$)/y;
  let pos = 0;
  while (pos < s.length) {
    re.lastIndex = pos;
    const m = re.exec(s);
    if (!m) {
      return null;
    }
    const parts = m[1].split(',').map((p) => p.trim());
    const trailingComma = parts.length > 1 && parts[parts.length - 1] === '';
    const dims = trailingComma ? parts.slice(0, -1) : parts;
    if (dims.length === 0 || dims.some((p) => !DIM.test(p)) || !dims.some((p) => /^[0-9]+$/.test(p))) {
      return null;
    }
    tuples.push(dims.length === 1 ? `(${dims[0]},)` : `(${dims.join(', ')})`);
    pos = re.lastIndex;
  }
  return tuples.length ? tuples.join(', ') : null;
}
