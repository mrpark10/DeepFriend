export interface Frame {
  file: string;
  line: number;
  name: string;
}

export interface ParsedError {
  type: string;
  message: string;
  frames: Frame[];
  raw: string;
}

const ANSI = /\u001b\[[0-9;]*[A-Za-z]/g;
const HEADER = /^Traceback \(most recent call last\):\s*$/;
const IPY_HEADER = /^(\S+)\s+Traceback \(most recent call last\)\s*$/;
const FRAME = /^\s*File "(.+)", line (\d+)(?:, in (.+))?\s*$/;
const IPY_FILE = /^\s*File (.+):(\d+), in (.+)$/;
const IPY_CELL = /^\s*Cell In\[(\d+)\], line (\d+)\s*$/;
const ERROR_LINE = /^((?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Exit|Interrupt|Warning|Iteration))(?::\s?(.*))?$/;
const CHAIN = /^(During handling of the above exception, another exception occurred:|The above exception was the direct cause of the following exception:)\s*$/;

export function stripAnsi(text: string): string {
  return text.replace(ANSI, '');
}

/** Extract Python tracebacks (standard, SyntaxError-style and IPython-style) from terminal output. */
export function findTracebacks(output: string): ParsedError[] {
  const lines = stripAnsi(output).replace(/\r\n?/g, '\n').split('\n');
  const found: ParsedError[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (HEADER.test(line) || IPY_HEADER.test(line)) {
      const r = parseBlock(lines, i + 1);
      if (!r) {
        i++;
        continue;
      }
      const { err, next } = r;
      // a chained exception replaces the previous one
      let j = next;
      while (j < lines.length && lines[j].trim() === '') {
        j++;
      }
      if (j < lines.length && CHAIN.test(lines[j])) {
        i = j + 1;
        continue;
      }
      found.push(err);
      i = next;
      continue;
    }
    const fm = FRAME.exec(line);
    if (fm && fm[3] === undefined) {
      // SyntaxError / IndentationError are printed without a header
      for (let k = i + 1; k < Math.min(lines.length, i + 6); k++) {
        const em = ERROR_LINE.exec(lines[k]);
        if (em && /^(SyntaxError|IndentationError|TabError)$/.test(em[1])) {
          found.push({
            type: em[1],
            message: (em[2] ?? '').trim(),
            frames: [{ file: fm[1], line: parseInt(fm[2], 10), name: '' }],
            raw: lines.slice(i, k + 1).join('\n'),
          });
          i = k + 1;
          break;
        }
      }
      i++;
      continue;
    }
    i++;
  }
  return found;
}

function parseBlock(lines: string[], start: number): { err: ParsedError; next: number } | null {
  const frames: Frame[] = [];
  let i = start;
  for (; i < lines.length; i++) {
    const line = lines[i];
    const fm = FRAME.exec(line);
    if (fm) {
      frames.push({ file: fm[1], line: parseInt(fm[2], 10), name: (fm[3] ?? '').trim() });
      continue;
    }
    const cm = IPY_CELL.exec(line);
    if (cm) {
      frames.push({ file: `Cell In[${cm[1]}]`, line: parseInt(cm[2], 10), name: '' });
      continue;
    }
    const im = IPY_FILE.exec(line);
    if (im) {
      frames.push({ file: im[1].trim(), line: parseInt(im[2], 10), name: im[3].trim() });
      continue;
    }
    if (/^\S/.test(line) && !/^-{5,}/.test(line)) {
      const em = ERROR_LINE.exec(line);
      if (em) {
        const msg: string[] = [(em[2] ?? '').trim()];
        let k = i + 1;
        // multi-line messages (Keras) continue until the next traceback or end of output
        while (k < lines.length && !HEADER.test(lines[k]) && !IPY_HEADER.test(lines[k]) && !CHAIN.test(lines[k]) && msg.length < 40) {
          if (FRAME.test(lines[k]) && /^\s*File "/.test(lines[k])) {
            break;
          }
          if (lines[k].trim() === '' && (k + 1 >= lines.length || lines[k + 1].trim() === '' || /^[A-Za-z].*[:>$#]\s*$/.test(lines[k + 1]))) {
            break;
          }
          msg.push(lines[k]);
          k++;
        }
        const message = msg.join('\n').trim();
        return { err: { type: em[1], message, frames, raw: lines.slice(start - 1, k).join('\n') }, next: k };
      }
    }
  }
  return null;
}

const LIB = /[\\/](site-packages|dist-packages)[\\/]|[\\/]lib[\\/]python\d|[\\/]Lib[\\/](?!site-packages)|^<frozen|^<string>|^<stdin>/;

function norm(p: string): string {
  return p.replace(/\\/g, '/').toLowerCase().replace(/\/+$/, '');
}

/** The frame the user most likely wants to jump to: last frame of their own code. */
export function userFrame(err: ParsedError, workspaceFolders: string[]): Frame | undefined {
  const folders = workspaceFolders.map(norm);
  const candidates = err.frames.filter((f) => !LIB.test(f.file));
  if (folders.length) {
    const inside = candidates.filter((f) => folders.some((w) => norm(f.file).startsWith(w + '/')));
    if (inside.length) {
      return inside[inside.length - 1];
    }
  }
  return candidates.length ? candidates[candidates.length - 1] : undefined;
}
