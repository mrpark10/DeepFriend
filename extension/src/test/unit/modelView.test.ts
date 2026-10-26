import { describe, expect, it } from 'vitest';
import { formatBytes, formatCount, pickModel } from '../../f4/modelData';
import type { ModelInfo } from '../../server/types';

const m = (name: string, line: number, endLine: number): ModelInfo => ({ name, line, endLine } as ModelInfo);

describe('model view data', () => {
  it('picks the model under the cursor, else the first one', () => {
    const models = [m('A', 2, 10), m('B', 12, 30)];
    expect(pickModel(models, 15)?.name).toBe('B');
    expect(pickModel(models, 5)?.name).toBe('A');
    expect(pickModel(models, 40)?.name).toBe('A');
    expect(pickModel([], 3)).toBeUndefined();
  });

  it('prefers a model with blocks when the cursor is outside every class', () => {
    const a = { ...m('Block', 2, 8), blocks: [] } as ModelInfo;
    const b = { ...m('Net', 10, 30), blocks: [{ name: 'fc' }] } as unknown as ModelInfo;
    expect(pickModel([a, b], 50)?.name).toBe('Net');
  });

  it('formats parameter counts', () => {
    expect(formatCount(950)).toBe('950');
    expect(formatCount(61706)).toBe('61.7K');
    expect(formatCount(11_689_512)).toBe('11.7M');
    expect(formatCount(1_500_000_000)).toBe('1.5B');
    expect(formatCount(null)).toBe('?');
  });

  it('formats bytes', () => {
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(1_019_672)).toBe('1.0 MB');
    expect(formatBytes(3.2e9)).toBe('3.2 GB');
  });
});
