import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { describe, expect, it } from 'vitest';
import { codeFiles, configDiff, headlineMetric } from '../../f5/compareData';
import type { RunMeta } from '../../f2/runStore';

const meta = (o: Partial<RunMeta>): RunMeta => ({ id: 'x', startedAt: '2026-10-26T10:00:00+09:00', status: 'finished', config: {}, numClasses: null, ...o }) as RunMeta;

describe('configDiff', () => {
  it('lists differing settings first, then the same ones, alphabetically', () => {
    const a = meta({ config: { lr: 0.01, batch_size: 32, optimizer: 'Adam' }, model: { params: 100 }, numClasses: 10 });
    const b = meta({ config: { lr: 0.001, batch_size: 32, epochs: 5 }, model: { params: 100 }, numClasses: 10 });
    const rows = configDiff(a, b);
    expect(rows.filter((r) => !r.same).map((r) => r.key)).toEqual(['epochs', 'lr', 'optimizer']);
    expect(rows.find((r) => r.key === 'lr')).toEqual({ key: 'lr', a: '0.01', b: '0.001', same: false });
    expect(rows.find((r) => r.key === 'epochs')).toEqual({ key: 'epochs', a: '-', b: '5', same: false });
    expect(rows.find((r) => r.key === 'model.params')?.same).toBe(true);
    expect(rows.find((r) => r.key === 'numClasses')?.same).toBe(true);
  });

  it('includes final results', () => {
    const rows = configDiff(meta({ final: { val_acc: 0.91 } }), meta({ final: { val_acc: 0.87 } }));
    expect(rows.find((r) => r.key === 'final.val_acc')).toEqual({ key: 'final.val_acc', a: '0.91', b: '0.87', same: false });
  });
});

describe('headlineMetric', () => {
  it('prefers validation accuracy, then accuracy, then losses', () => {
    expect(headlineMetric(meta({ final: { val_acc: 0.912, acc: 0.99, loss: 0.1 } }))).toEqual({ key: 'val_acc', text: '91.2%' });
    expect(headlineMetric(meta({ final: { acc: 0.5 } }))).toEqual({ key: 'acc', text: '50.0%' });
    expect(headlineMetric(meta({ final: { val_loss: 0.1234, loss: 0.3 } }))).toEqual({ key: 'val_loss', text: '0.123' });
    expect(headlineMetric(meta({ final: {} }))).toBeNull();
  });
});

describe('codeFiles', () => {
  it('returns the union of snapshot files, sorted', () => {
    const a = fs.mkdtempSync(path.join(os.tmpdir(), 'dfA-'));
    const b = fs.mkdtempSync(path.join(os.tmpdir(), 'dfB-'));
    fs.mkdirSync(path.join(a, 'code', 'models'), { recursive: true });
    fs.writeFileSync(path.join(a, 'code', 'train.py'), 'a');
    fs.writeFileSync(path.join(a, 'code', 'models', 'net.py'), 'a');
    fs.mkdirSync(path.join(b, 'code'), { recursive: true });
    fs.writeFileSync(path.join(b, 'code', 'train.py'), 'b');
    fs.writeFileSync(path.join(b, 'code', 'utils.py'), 'b');
    expect(codeFiles(a, b)).toEqual(['models/net.py', 'train.py', 'utils.py']);
    expect(codeFiles(a, path.join(b, 'missing'))).toEqual(['models/net.py', 'train.py']);
  });
});
