import * as fs from 'node:fs';
import * as path from 'node:path';
import { describe, expect, it } from 'vitest';
import { findTracebacks, userFrame } from '../../f3/traceback';

const FIX = path.join(__dirname, '..', 'fixtures', 'errors');
const read = (name: string) => fs.readFileSync(path.join(FIX, name), 'utf8');

describe('findTracebacks', () => {
  it('parses a normal PyTorch traceback', () => {
    const [e] = findTracebacks(read('shape_matmul.txt'));
    expect(e.type).toBe('RuntimeError');
    expect(e.message).toBe('mat1 and mat2 shapes cannot be multiplied (32x12544 and 10000x10)');
    expect(e.frames[0]).toEqual({ file: '/home/student/project/train.py', line: 6, name: '<module>' });
    expect(e.frames.length).toBeGreaterThan(1);
  });

  it('keeps only the final exception of a chain', () => {
    const errs = findTracebacks(read('chained.txt'));
    expect(errs).toHaveLength(1);
    expect(errs[0].type).toBe('ValueError');
    expect(errs[0].message).toBe('bad config');
    expect(errs[0].frames[0].line).toBe(4);
  });

  it('parses SyntaxError without a Traceback header', () => {
    const [e] = findTracebacks(read('syntax_error.txt'));
    expect(e.type).toBe('SyntaxError');
    expect(e.frames).toEqual([{ file: '/home/student/project/train.py', line: 1, name: '' }]);
  });

  it('keeps multi-line messages and strips ANSI codes (Keras)', () => {
    const [e] = findTracebacks(read('keras_input.txt'));
    expect(e.type).toBe('ValueError');
    expect(e.message).toContain('Expected shape (None, 784), but input has incompatible shape (2, 28, 28)');
    expect(e.message).not.toContain('\u001b');
  });

  it('parses dotted exception types', () => {
    const [e] = findTracebacks(read('cuda_oom.synthetic.txt'));
    expect(e.type).toBe('torch.OutOfMemoryError');
    expect(e.message.startsWith('CUDA out of memory.')).toBe(true);
  });

  it('parses windows traceback with korean path', () => {
    const [e] = findTracebacks(read('windows_korean.synthetic.txt'));
    expect(e.frames[0]).toEqual({ file: 'C:\\Users\\박준호\\딥러닝 과제\\train.py', line: 12, name: '<module>' });
  });

  it('parses Jupyter/IPython style output', () => {
    const [e] = findTracebacks(read('jupyter.synthetic.txt'));
    expect(e.type).toBe('RuntimeError');
    expect(e.message).toBe('mat1 and mat2 shapes cannot be multiplied (1x10 and 20x5)');
    expect(e.frames[0]).toEqual({ file: 'Cell In[3]', line: 5, name: '' });
    expect(e.frames[1].line).toBe(1501);
  });

  it('handles CRLF and finds two separate errors in one output', () => {
    const text = (read('name_error.txt') + '\nsome output\n' + read('zero_division.txt')).replace(/\n/g, '\r\n');
    const errs = findTracebacks(text);
    expect(errs.map((e) => e.type)).toEqual(['NameError', 'ZeroDivisionError']);
  });

  it('returns nothing for normal output', () => {
    expect(findTracebacks('epoch 1 loss 0.3\nTraceback is a word\n')).toEqual([]);
  });

  it('ignores an incomplete traceback that is still streaming', () => {
    expect(findTracebacks('Traceback (most recent call last):\n  File "a.py", line 3, in <module>\n')).toEqual([]);
  });
});

describe('userFrame', () => {
  it('picks the last frame inside the workspace, skipping libraries', () => {
    const [e] = findTracebacks(read('shape_matmul.txt'));
    expect(userFrame(e, ['/home/student/project'])?.line).toBe(6);
  });

  it('works with Windows paths regardless of case', () => {
    const [e] = findTracebacks(read('windows_korean.synthetic.txt'));
    expect(userFrame(e, ['c:\\users\\박준호\\딥러닝 과제'])?.line).toBe(12);
  });

  it('falls back to the last non-library frame without a workspace', () => {
    const [e] = findTracebacks(read('cuda_oom.synthetic.txt'));
    expect(userFrame(e, [])?.file).toBe('/home/student/project/train.py');
  });

  it('returns undefined when every frame is library code', () => {
    const e = { type: 'X', message: '', raw: '', frames: [{ file: '/usr/lib/python3.11/site-packages/x.py', line: 1, name: 'f' }] };
    expect(userFrame(e, [])).toBeUndefined();
  });
});
