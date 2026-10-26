import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { RunStore } from '../../f2/runStore';

let tmp: string;
const meta = (o: Record<string, unknown> = {}) => JSON.stringify({ id: 'r1', status: 'running', pid: 1, startedAt: '2026-10-26T10:00:00+09:00', config: {}, numClasses: 3, deepfriendVersion: '0.1.0', ...o });

beforeEach(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'df-run-'));
});
afterEach(() => fs.rmSync(tmp, { recursive: true, force: true }));

describe('RunStore', () => {
  it('loads a run and reads new metrics incrementally', () => {
    fs.writeFileSync(path.join(tmp, 'meta.json'), meta());
    fs.writeFileSync(path.join(tmp, 'metrics.jsonl'), '{"step":1,"epoch":0,"split":"train","loss":2}\n');
    const store = new RunStore();
    const r = store.refresh(tmp)!;
    expect(r.meta.id).toBe('r1');
    expect(r.series.train.length).toBe(1);
    expect(r.series.numClasses).toBe(3);
    fs.appendFileSync(path.join(tmp, 'metrics.jsonl'), '{"step":2,"epoch":0,"split":"train","loss":1.5}\n{"step":3,"ep');
    expect(store.refresh(tmp)!.series.train.map((p) => p.step)).toEqual([1, 2]);
    fs.appendFileSync(path.join(tmp, 'metrics.jsonl'), 'och":0,"split":"train","loss":1.2}\n');
    expect(store.refresh(tmp)!.series.train.map((p) => p.step)).toEqual([1, 2, 3]);
  });

  it('returns null while meta.json is missing or half-written', () => {
    const store = new RunStore();
    expect(store.refresh(tmp)).toBeNull();
    fs.writeFileSync(path.join(tmp, 'meta.json'), '{"id": "r1", "sta');
    expect(store.refresh(tmp)).toBeNull();
  });

  it('starts over when metrics.jsonl was truncated', () => {
    fs.writeFileSync(path.join(tmp, 'meta.json'), meta());
    fs.writeFileSync(path.join(tmp, 'metrics.jsonl'), '{"step":1,"epoch":0,"split":"train","loss":2}\n{"step":2,"epoch":0,"split":"train","loss":2}\n');
    const store = new RunStore();
    store.refresh(tmp);
    fs.writeFileSync(path.join(tmp, 'metrics.jsonl'), '{"step":1,"epoch":0,"split":"train","loss":9}\n');
    expect(store.refresh(tmp)!.series.train.map((p) => p.loss)).toEqual([9]);
  });

  it('marks a running run whose process is gone as stopped', () => {
    fs.writeFileSync(path.join(tmp, 'meta.json'), meta({ pid: 999999 }));
    const store = new RunStore(() => false);
    const r = store.refresh(tmp)!;
    expect(r.meta.status).toBe('stopped');
    expect(JSON.parse(fs.readFileSync(path.join(tmp, 'meta.json'), 'utf8')).status).toBe('stopped');
  });

  it('keeps a running run running while its process lives', () => {
    fs.writeFileSync(path.join(tmp, 'meta.json'), meta());
    const store = new RunStore(() => true);
    expect(store.refresh(tmp)!.meta.status).toBe('running');
  });

  it('loads samples when present', () => {
    fs.writeFileSync(path.join(tmp, 'meta.json'), meta({ status: 'finished' }));
    fs.writeFileSync(path.join(tmp, 'samples.json'), JSON.stringify({ items: [{ shape: [2], pixels: [0, 255], label: 1 }] }));
    expect(new RunStore().refresh(tmp)!.samples?.items[0].label).toBe(1);
  });
});
