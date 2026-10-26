import * as fs from 'node:fs';
import * as path from 'node:path';
import type { RunSeries } from './doctorRules';
import { parseMetrics } from './runReader';

export interface RunMeta {
  id: string;
  name?: string | null;
  script?: string | null;
  cwd?: string;
  startedAt: string;
  endedAt?: string | null;
  status: 'running' | 'finished' | 'failed' | 'stopped';
  pid?: number;
  framework?: 'pytorch' | 'tensorflow' | null;
  config: Record<string, unknown>;
  model?: { name?: string | null; params?: number };
  numClasses: number | null;
  final?: Record<string, number>;
  best?: Record<string, number>;
  starred?: boolean;
  deepfriendVersion?: string;
  git?: string | null;
  record?: boolean;
}

export interface Samples {
  items: { shape: number[]; pixels: number[]; label: string | number | null }[];
}

export interface RunData {
  dir: string;
  meta: RunMeta;
  series: RunSeries;
  samples?: Samples;
}

function defaultAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e) {
    return (e as NodeJS.ErrnoException).code === 'EPERM';
  }
}

/** Reads run folders incrementally (only the new part of metrics.jsonl on each refresh). */
export class RunStore {
  private runs = new Map<string, RunData & { offset: number; size: number }>();

  constructor(private alive: (pid: number) => boolean = defaultAlive) {}

  get(dir: string): RunData | undefined {
    return this.runs.get(dir);
  }

  all(): RunData[] {
    return [...this.runs.values()];
  }

  forget(dir: string): void {
    this.runs.delete(dir);
  }

  refresh(dir: string): RunData | null {
    let meta: RunMeta;
    try {
      meta = JSON.parse(fs.readFileSync(path.join(dir, 'meta.json'), 'utf8')) as RunMeta;
    } catch {
      return this.runs.get(dir) ?? null;
    }
    if (meta.status === 'running' && typeof meta.pid === 'number' && !this.alive(meta.pid)) {
      meta.status = 'stopped';
      meta.endedAt = meta.endedAt ?? new Date().toISOString();
      try {
        const tmp = path.join(dir, 'meta.json.tmp');
        fs.writeFileSync(tmp, JSON.stringify(meta, null, 2));
        fs.renameSync(tmp, path.join(dir, 'meta.json'));
      } catch {
        // read-only folder: keep the in-memory status
      }
    }
    let entry = this.runs.get(dir);
    if (!entry) {
      entry = { dir, meta, series: { train: [], epochs: [], numClasses: null }, offset: 0, size: 0 };
      this.runs.set(dir, entry);
    }
    entry.meta = meta;
    entry.series.numClasses = meta.numClasses ?? null;
    const file = path.join(dir, 'metrics.jsonl');
    try {
      const size = fs.statSync(file).size;
      if (size < entry.offset) {
        entry.offset = 0;
        entry.series = { train: [], epochs: [], numClasses: meta.numClasses ?? null };
      }
      if (size > entry.offset) {
        const fd = fs.openSync(file, 'r');
        try {
          const buf = Buffer.alloc(size - entry.offset);
          fs.readSync(fd, buf, 0, buf.length, entry.offset);
          const lastNl = buf.lastIndexOf(0x0a);
          if (lastNl >= 0) {
            parseMetrics(buf.subarray(0, lastNl + 1).toString('utf8'), entry.series);
            entry.offset += lastNl + 1;
          }
        } finally {
          fs.closeSync(fd);
        }
      }
    } catch {
      // no metrics yet
    }
    if (!entry.samples) {
      try {
        entry.samples = JSON.parse(fs.readFileSync(path.join(dir, 'samples.json'), 'utf8')) as Samples;
      } catch {
        // none
      }
    }
    return entry;
  }
}
