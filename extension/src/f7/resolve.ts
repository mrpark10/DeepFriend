import { CONCEPTS } from './concepts';

export interface Resolved {
  key: string;
  param?: string;
}

const IDENT = /[A-Za-z0-9_]/;
const OPTIMIZERS = new Set(['SGD', 'Adam', 'AdamW']);

/** True when `offset` sits inside a string literal or a comment on its line. */
function inStringOrComment(text: string, offset: number): boolean {
  const lineStart = text.lastIndexOf('\n', offset - 1) + 1;
  let quote: string | null = null;
  for (let i = lineStart; i < offset; i++) {
    const c = text[i];
    if (quote) {
      if (c === '\\') {
        i++;
      } else if (c === quote) {
        quote = null;
      }
    } else if (c === '#') {
      return true;
    } else if (c === '"' || c === "'") {
      quote = c;
    }
  }
  return quote !== null;
}

function importSource(text: string, name: string): string | null {
  const re = new RegExp(`^\\s*from\\s+([\\w.]+)\\s+import\\s+[^\\n]*\\b${name}\\b`, 'm');
  const m = re.exec(text);
  return m ? m[1] : null;
}

/** Map a dotted name like `nn.Conv2d` or `layers.Dense` to a concept key. */
export function resolveName(dotted: string, text: string): string | null {
  const parts = dotted.split('.');
  const last = parts[parts.length - 1];
  let fw: 'torch' | 'keras' | null = null;
  let viaOptim = parts.includes('optim');
  if (parts.length > 1) {
    if (parts.includes('keras') || parts[0] === 'layers' || parts[0] === 'tf') {
      fw = 'keras';
    } else if (['nn', 'torch', 'optim', 'F'].includes(parts[0])) {
      fw = 'torch';
    }
  } else {
    const src = importSource(text, last);
    if (src) {
      if (src.includes('keras')) {
        fw = 'keras';
      } else if (src.startsWith('torch')) {
        fw = 'torch';
        viaOptim = viaOptim || src.includes('optim');
      }
    }
    if (!fw) {
      const hasTorch = /^\s*(?:import|from)\s+torch\b/m.test(text);
      const hasKeras = /^\s*(?:import|from)\s+(?:tensorflow|keras)\b/m.test(text);
      fw = hasKeras && !hasTorch ? 'keras' : 'torch';
    }
  }
  if (!fw) {
    return null;
  }
  let key: string;
  if (fw === 'torch' && OPTIMIZERS.has(last) && viaOptim) {
    key = 'torch.optim.' + last;
  } else {
    key = fw + '.' + last;
  }
  return key in CONCEPTS ? key : null;
}

/** Find the concept (and keyword parameter) under `offset` in `text`. */
export function resolveAt(text: string, offset: number): Resolved | null {
  if (offset < 0 || offset > text.length || inStringOrComment(text, offset)) {
    return null;
  }
  let ws = offset;
  while (ws > 0 && IDENT.test(text[ws - 1])) {
    ws--;
  }
  let we = offset;
  while (we < text.length && IDENT.test(text[we])) {
    we++;
  }
  if (ws === we) {
    return null;
  }
  const word = text.slice(ws, we);
  // keyword argument? `name=` but not `name==`
  let k = we;
  while (k < text.length && (text[k] === ' ' || text[k] === '\t')) {
    k++;
  }
  if (text[k] === '=' && text[k + 1] !== '=') {
    const callee = enclosingCallee(text, ws);
    if (!callee) {
      return null;
    }
    const key = resolveName(callee, text);
    return key ? { key, param: word } : null;
  }
  // dotted name around the cursor
  let ds = ws;
  while (ds > 0 && (IDENT.test(text[ds - 1]) || text[ds - 1] === '.')) {
    ds--;
  }
  let de = we;
  while (de < text.length && (IDENT.test(text[de]) || (text[de] === '.' && IDENT.test(text[de + 1] ?? '')))) {
    de++;
  }
  const dotted = text.slice(ds, de).replace(/^\.+/, '');
  const key = resolveName(dotted, text);
  return key ? { key } : null;
}

/** Dotted name of the call whose parentheses enclose `pos`, e.g. `nn.Conv2d`. */
function enclosingCallee(text: string, pos: number): string | null {
  let depth = 0;
  for (let i = pos - 1; i >= 0; i--) {
    const c = text[i];
    if (c === ')' || c === ']' || c === '}') {
      depth++;
    } else if (c === '(' || c === '[' || c === '{') {
      if (depth === 0) {
        if (c !== '(') {
          return null;
        }
        let e = i;
        while (e > 0 && text[e - 1] === ' ') {
          e--;
        }
        let s = e;
        while (s > 0 && (IDENT.test(text[s - 1]) || text[s - 1] === '.')) {
          s--;
        }
        return s < e ? text.slice(s, e) : null;
      }
      depth--;
    }
  }
  return null;
}
