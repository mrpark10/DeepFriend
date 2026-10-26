import { describe, expect, it } from 'vitest';
import { buildChart } from '../../f2/chart';

describe('buildChart', () => {
  it('aligns validation points to the last step of their epoch', () => {
    const c = buildChart({
      train: [
        { step: 1, loss: 2, epoch: 0, acc: 0.1 },
        { step: 2, loss: 1.5, epoch: 0, acc: 0.2 },
        { step: 3, loss: 1, epoch: 1, acc: 0.3 },
        { step: 4, loss: null, nan: true, epoch: 1 },
      ],
      epochs: [{ epoch: 0, val_loss: 1.8, val_acc: 0.15 }, { epoch: 1, val_loss: 1.2, val_acc: 0.25 }],
      numClasses: 3,
    });
    expect(c.x).toEqual([1, 2, 3, 4]);
    expect(c.trainLoss).toEqual([2, 1.5, 1, null]);
    expect(c.valLoss).toEqual([null, 1.8, null, 1.2]);
    expect(c.valAcc).toEqual([null, 0.15, null, 0.25]);
    expect(c.hasAcc).toBe(true);
  });

  it('decimates long runs to at most 2000 points by averaging buckets', () => {
    const train = Array.from({ length: 10_000 }, (_, i) => ({ step: i + 1, loss: 1, epoch: Math.floor(i / 1000) }));
    const c = buildChart({ train, epochs: [{ epoch: 9, val_loss: 0.5 }], numClasses: null });
    expect(c.x.length).toBeLessThanOrEqual(2001);
    expect(c.trainLoss.every((v) => v === 1 || v === null)).toBe(true);
    expect(c.valLoss.filter((v) => v !== null)).toEqual([0.5]);
    expect(c.hasAcc).toBe(false);
  });

  it('plots epoch-only runs (manual val logging) on an epoch axis', () => {
    const c = buildChart({ train: [], epochs: [{ epoch: 0, val_loss: 1 }, { epoch: 1, val_loss: 0.8 }], numClasses: null });
    expect(c.x).toEqual([1, 2]);
    expect(c.valLoss).toEqual([1, 0.8]);
  });
});
