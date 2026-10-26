import * as fs from 'node:fs';
import * as path from 'node:path';
import { describe, expect, it } from 'vitest';
import { diagnose, RunSeries } from '../../f2/doctorRules';
import { parseMetrics } from '../../f2/runReader';

const FIX = path.join(__dirname, '..', 'fixtures', 'runs');

function load(name: string): RunSeries {
  const meta = JSON.parse(fs.readFileSync(path.join(FIX, name, 'meta.json'), 'utf8'));
  const series = parseMetrics(fs.readFileSync(path.join(FIX, name, 'metrics.jsonl'), 'utf8'));
  series.numClasses = meta.numClasses;
  return series;
}

describe('parseMetrics', () => {
  it('splits train steps and per-epoch aggregates', () => {
    const s = load('healthy');
    expect(s.train.length).toBe(174);
    expect(s.train[0].step).toBe(1);
    expect(s.epochs.map((e) => e.epoch)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(typeof s.epochs[0].loss).toBe('number');
    expect(typeof s.epochs[0].val_loss).toBe('number');
    expect(typeof s.epochs[0].val_acc).toBe('number');
  });

  it('keeps NaN steps as null with a flag', () => {
    const s = load('nan');
    expect(s.train.some((p) => p.loss === null && p.nan)).toBe(true);
  });

  it('ignores a truncated last line and garbage lines', () => {
    const s = parseMetrics('{"step":1,"epoch":0,"split":"train","loss":1.0}\nnot json\n{"step":2,"epoch":0,"split":"tr');
    expect(s.train.map((p) => p.step)).toEqual([1]);
  });

  it('reads user-logged validation values', () => {
    const s = parseMetrics('{"step":0,"epoch":3,"split":"val","agg":true,"loss":0.5,"acc":0.8}\n');
    expect(s.epochs).toEqual([{ epoch: 3, val_loss: 0.5, val_acc: 0.8 }]);
  });
});

describe('diagnose (calibration examples)', () => {
  const cases: [string, string][] = [
    ['nan', 'nan'],
    ['overfit', 'overfit'],
    ['plateau', 'plateau'],
    ['oscillation', 'oscillation'],
    ['chance', 'chance'],
  ];
  for (const [fixture, id] of cases) {
    it(`${fixture} → ${id} first`, () => {
      const f = diagnose(load(fixture));
      expect(f.length).toBeGreaterThan(0);
      expect(f[0].id).toBe(id);
    });
  }

  it('finds nothing wrong with healthy training', () => {
    expect(diagnose(load('healthy'))).toEqual([]);
  });

  it('only reports NaN when the loss is NaN (other rules are meaningless then)', () => {
    expect(diagnose(load('nan')).map((f) => f.id)).toEqual(['nan']);
  });

  it('suggests concrete actions', () => {
    expect(diagnose(load('nan'))[0].actions).toContainEqual({ kind: 'setLr', factor: 0.1 });
    expect(diagnose(load('plateau'))[0].actions).toContainEqual({ kind: 'setLr', factor: 10 });
    expect(diagnose(load('oscillation'))[0].actions).toEqual(expect.arrayContaining([{ kind: 'setLr', factor: 0.1 }, { kind: 'setBatch', factor: 2 }]));
    expect(diagnose(load('chance'))[0].actions).toContainEqual({ kind: 'showSamples' });
    expect(diagnose(load('overfit'))[0].actions).toContainEqual({ kind: 'snippet', id: 'earlyStopping' });
  });

  it('gives the numbers used in the explanation', () => {
    const o = diagnose(load('overfit'))[0];
    expect(o.args.bestEpoch).toBe(1); // shown 1-based
    expect(o.args.risePct).toBeGreaterThan(10);
    const c = diagnose(load('chance'))[0];
    expect(c.args.chancePct).toBe(10);
  });
});

describe('diagnose (edge cases)', () => {
  const empty: RunSeries = { train: [], epochs: [], numClasses: null };

  it('empty and tiny series give nothing', () => {
    expect(diagnose(empty)).toEqual([]);
    expect(diagnose({ ...empty, train: [{ step: 1, loss: 2 }] })).toEqual([]);
  });

  it('all-NaN gives only nan', () => {
    const train = Array.from({ length: 10 }, (_, i) => ({ step: i + 1, loss: null, nan: true }));
    expect(diagnose({ ...empty, train }).map((f) => f.id)).toEqual(['nan']);
  });

  it('no validation data → never overfit', () => {
    const train = Array.from({ length: 300 }, (_, i) => ({ step: i + 1, loss: 2 * Math.exp(-i / 50) + 0.01 }));
    expect(diagnose({ ...empty, train }).map((f) => f.id)).not.toContain('overfit');
  });

  it('a converged run at near-zero loss is not a plateau', () => {
    const train = [{ step: 1, loss: 2.3 }, ...Array.from({ length: 200 }, (_, i) => ({ step: i + 2, loss: 0.001 }))];
    expect(diagnose({ ...empty, train })).toEqual([]);
  });

  it('chance needs the number of classes', () => {
    const epochs = [0, 1, 2, 3].map((e) => ({ epoch: e, loss: 2.3, acc: 0.1, val_loss: 2.3 - e * 0.1, val_acc: 0.1 }));
    expect(diagnose({ train: [], epochs, numClasses: null }).map((f) => f.id)).not.toContain('chance');
    expect(diagnose({ train: [], epochs, numClasses: 10 }).map((f) => f.id)).toContain('chance');
  });

  it('a short run (1 epoch, few steps) is not judged yet', () => {
    const train = Array.from({ length: 20 }, (_, i) => ({ step: i + 1, loss: 1.5 }));
    expect(diagnose({ train, epochs: [], numClasses: 4 })).toEqual([]);
  });
});

describe('diagnose (oscillation severity)', () => {
  it('very large spikes count as oscillation even with a downward trend', () => {
    // starts calm at 1.0, then jumps between 3.0 and 0.1 while slowly trending down
    const train = Array.from({ length: 200 }, (_, i) => ({ step: i + 1, loss: i < 10 ? 1.0 : (1 - i * 0.002) * (i % 2 ? 0.1 : 3.0) }));
    expect(diagnose({ train, epochs: [], numClasses: null }).map((f) => f.id)).toContain('oscillation');
  });
});
