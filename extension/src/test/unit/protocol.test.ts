import { describe, expect, it } from 'vitest';
import { LineSplitter } from '../../server/protocol';

describe('LineSplitter', () => {
  it('splits lines across chunks', () => {
    const s = new LineSplitter();
    expect(s.push(Buffer.from('{"a":1}\n{"b"'))).toEqual(['{"a":1}']);
    expect(s.push(Buffer.from(':2}\n'))).toEqual(['{"b":2}']);
  });

  it('handles a UTF-8 character split between chunks', () => {
    const s = new LineSplitter();
    const bytes = Buffer.from('{"t":"박준호"}\n', 'utf8');
    const out = [...s.push(bytes.subarray(0, 8)), ...s.push(bytes.subarray(8))];
    expect(out).toEqual(['{"t":"박준호"}']);
  });

  it('strips CR and skips blank lines', () => {
    const s = new LineSplitter();
    expect(s.push(Buffer.from('a\r\n\r\nb\n'))).toEqual(['a', 'b']);
  });
});
