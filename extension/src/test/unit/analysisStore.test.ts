import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AnalysisStore, DocLike } from '../../f1/analysisStore';
import type { AnalysisResult } from '../../server/types';

function doc(text: string, version = 1, uri = 'file:///a.py', languageId = 'python'): DocLike {
  return { uri: { toString: () => uri }, version, languageId, getText: () => text };
}

const TORCH = 'import torch\n';
const result = (n: number): AnalysisResult => ({ framework: 'pytorch', models: [{ name: 'M' + n } as never], syntaxError: null });

describe('AnalysisStore', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('analyzes deep-learning files on open', async () => {
    const req = vi.fn(async () => result(1));
    const s = new AnalysisStore(req);
    s.onOpen(doc(TORCH));
    await vi.runAllTimersAsync();
    expect(req).toHaveBeenCalledTimes(1);
    expect(s.get('file:///a.py')?.models[0].name).toBe('M1');
  });

  it('ignores files without torch/tensorflow imports and non-python files', async () => {
    const req = vi.fn(async () => result(1));
    const s = new AnalysisStore(req);
    s.onOpen(doc('print(1)\n'));
    s.onOpen(doc(TORCH, 1, 'file:///b.md', 'markdown'));
    await vi.runAllTimersAsync();
    expect(req).not.toHaveBeenCalled();
  });

  it('debounces typing by 400ms', async () => {
    const req = vi.fn(async () => result(1));
    const s = new AnalysisStore(req);
    s.onChange(doc(TORCH + 'a', 2));
    await vi.advanceTimersByTimeAsync(200);
    s.onChange(doc(TORCH + 'ab', 3));
    await vi.advanceTimersByTimeAsync(399);
    expect(req).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(2);
    expect(req).toHaveBeenCalledTimes(1);
    expect(req).toHaveBeenCalledWith('file:///a.py', TORCH + 'ab');
  });

  it('analyzes immediately on save and cancels the pending debounce', async () => {
    const req = vi.fn(async () => result(1));
    const s = new AnalysisStore(req);
    s.onChange(doc(TORCH + 'x', 2));
    s.onSave(doc(TORCH + 'x', 2));
    await vi.advanceTimersByTimeAsync(0);
    expect(req).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1000);
    expect(req).toHaveBeenCalledTimes(1);
  });

  it('drops a response for an outdated document version', async () => {
    let resolveFirst: (r: AnalysisResult) => void = () => {};
    const req = vi.fn()
      .mockImplementationOnce(() => new Promise<AnalysisResult>((r) => { resolveFirst = r; }))
      .mockImplementationOnce(async () => result(2));
    const s = new AnalysisStore(req);
    const changes: string[] = [];
    s.onDidChange((u) => changes.push(u));
    const d1 = doc(TORCH, 1);
    s.onSave(d1);
    await vi.advanceTimersByTimeAsync(0);
    const d2 = doc(TORCH + '#', 2);
    s.trackVersion(d2);
    resolveFirst(result(1));
    await vi.advanceTimersByTimeAsync(0);
    expect(s.get('file:///a.py')).toBeUndefined();
    s.onSave(d2);
    await vi.advanceTimersByTimeAsync(0);
    expect(s.get('file:///a.py')?.models[0].name).toBe('M2');
    expect(changes).toEqual(['file:///a.py']);
  });

  it('ignores null (cancelled) and failed responses but keeps the previous result', async () => {
    const req = vi.fn()
      .mockImplementationOnce(async () => result(1))
      .mockImplementationOnce(async () => null)
      .mockImplementationOnce(async () => { throw new Error('server down'); });
    const s = new AnalysisStore(req);
    s.onSave(doc(TORCH, 1));
    await vi.advanceTimersByTimeAsync(0);
    s.onSave(doc(TORCH, 2));
    await vi.advanceTimersByTimeAsync(0);
    s.onSave(doc(TORCH, 3));
    await vi.advanceTimersByTimeAsync(0);
    expect(s.get('file:///a.py')?.models[0].name).toBe('M1');
  });

  it('forgets closed documents', async () => {
    const s = new AnalysisStore(vi.fn(async () => result(1)));
    const d = doc(TORCH);
    s.onSave(d);
    await vi.advanceTimersByTimeAsync(0);
    s.onClose(d);
    expect(s.get('file:///a.py')).toBeUndefined();
  });

  it('clears results when imports are removed', async () => {
    const s = new AnalysisStore(vi.fn(async () => result(1)));
    s.onSave(doc(TORCH, 1));
    await vi.advanceTimersByTimeAsync(0);
    s.onSave(doc('print(1)\n', 2));
    await vi.advanceTimersByTimeAsync(0);
    expect(s.get('file:///a.py')).toBeUndefined();
  });
});
