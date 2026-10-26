import { Emitter } from '../server/client';
import { findTracebacks, ParsedError, stripAnsi } from './traceback';

export type ErrorSource = 'terminal' | 'notebook' | 'runFile';

export interface ReportedError {
  error: ParsedError;
  source: ErrorSource;
}

const DEDUP_MS = 10_000;

/** Collects errors from every source and drops duplicates seen within 10 seconds. */
export class ErrorHub {
  private recent = new Map<string, number>();
  private emitter = new Emitter<ReportedError>();
  readonly onError = this.emitter.event;

  report(error: ParsedError, source: ErrorSource): void {
    const f = error.frames[error.frames.length - 1];
    const key = `${error.type}|${error.message.split('\n')[0]}|${f ? f.file + ':' + f.line : ''}`;
    const now = Date.now();
    for (const [k, t] of this.recent) {
      if (now - t > DEDUP_MS) {
        this.recent.delete(k);
      }
    }
    if (this.recent.has(key)) {
      return;
    }
    this.recent.set(key, now);
    this.emitter.fire({ error, source });
  }
}

const MAX = 200_000;

/** Accumulates one terminal command's output and yields each traceback once it is complete. */
export class TerminalBuffer {
  private buf = '';

  push(chunk: string): ParsedError[] {
    this.buf += stripAnsi(chunk).replace(/\r\n?/g, '\n');
    if (this.buf.length > MAX) {
      this.buf = this.buf.slice(this.buf.length - MAX);
    }
    const errs = findTracebacks(this.buf);
    if (!errs.length) {
      return [];
    }
    const last = errs[errs.length - 1];
    const at = this.buf.lastIndexOf(last.raw);
    this.buf = at >= 0 ? this.buf.slice(at + last.raw.length) : '';
    return errs;
  }

  size(): number {
    return this.buf.length;
  }
}
