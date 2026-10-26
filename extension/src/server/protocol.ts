import { StringDecoder } from 'node:string_decoder';

/** Splits a byte stream into UTF-8 text lines, safe across chunk boundaries. */
export class LineSplitter {
  private decoder = new StringDecoder('utf8');
  private buf = '';

  push(chunk: Buffer): string[] {
    this.buf += this.decoder.write(chunk);
    const parts = this.buf.split('\n');
    this.buf = parts.pop() ?? '';
    return parts.map((l) => l.replace(/\r$/, '')).filter((l) => l.length > 0);
  }
}
