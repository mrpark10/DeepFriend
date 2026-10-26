import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ErrorHub, TerminalBuffer } from '../../f3/errorHub';
import type { ParsedError } from '../../f3/traceback';

const err = (msg: string, line = 3): ParsedError => ({ type: 'RuntimeError', message: msg, raw: '', frames: [{ file: '/p/train.py', line, name: '<module>' }] });

describe('ErrorHub', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('emits each new error once', () => {
    const hub = new ErrorHub();
    const seen: string[] = [];
    hub.onError((e) => seen.push(e.error.message + '@' + e.source));
    hub.report(err('a'), 'terminal');
    hub.report(err('b'), 'notebook');
    expect(seen).toEqual(['a@terminal', 'b@notebook']);
  });

  it('drops the same error from another source within 10 seconds', () => {
    const hub = new ErrorHub();
    const seen: string[] = [];
    hub.onError((e) => seen.push(e.source));
    hub.report(err('a'), 'terminal');
    hub.report(err('a'), 'runFile');
    vi.advanceTimersByTime(9000);
    hub.report(err('a'), 'terminal');
    expect(seen).toEqual(['terminal']);
    vi.advanceTimersByTime(11_000);
    hub.report(err('a'), 'terminal');
    expect(seen).toEqual(['terminal', 'terminal']);
  });

  it('treats the same message on another line as a different error', () => {
    const hub = new ErrorHub();
    const seen: number[] = [];
    hub.onError((e) => seen.push(e.error.frames[0].line));
    hub.report(err('a', 3), 'terminal');
    hub.report(err('a', 9), 'terminal');
    expect(seen).toEqual([3, 9]);
  });
});

describe('TerminalBuffer', () => {
  it('reports a traceback once it is complete, even if split across chunks', () => {
    const b = new TerminalBuffer();
    expect(b.push('epoch 1\nTraceback (most recent call last):\n  File "/p/a.py", line 2, in <module>\n')).toEqual([]);
    const out = b.push('    x()\nNameError: name \'x\' is not defined\n');
    expect(out.map((e) => e.type)).toEqual(['NameError']);
    expect(b.push('more output\n')).toEqual([]);
  });

  it('reports two errors in one execution separately', () => {
    const b = new TerminalBuffer();
    const t = (m: string) => `Traceback (most recent call last):\n  File "/p/a.py", line 1, in <module>\nValueError: ${m}\n`;
    expect(b.push(t('one')).length).toBe(1);
    expect(b.push('ok\n' + t('two')).map((e) => e.message)).toEqual(['two']);
  });

  it('caps memory at 200 KB', () => {
    const b = new TerminalBuffer();
    b.push('x'.repeat(500_000));
    expect(b.size()).toBeLessThanOrEqual(200_000);
  });
});
