import type { EpochPoint, RunSeries, TrainPoint } from './doctorRules';

interface Rec {
  step?: number;
  epoch?: number;
  split?: string;
  agg?: boolean;
  loss?: number | null;
  acc?: number;
  nan?: boolean;
  lr?: number;
}

/** Parse metrics.jsonl text into a series. A truncated last line or garbage line is skipped. */
export function parseMetrics(text: string, into?: RunSeries): RunSeries {
  const series: RunSeries = into ?? { train: [], epochs: [], numClasses: null };
  const byEpoch = new Map<number, EpochPoint>(series.epochs.map((e) => [e.epoch, e]));
  for (const line of text.split('\n')) {
    if (!line.trim()) {
      continue;
    }
    let r: Rec;
    try {
      r = JSON.parse(line) as Rec;
    } catch {
      continue;
    }
    if (r.split === 'train' && !r.agg) {
      const p: TrainPoint = { step: r.step ?? series.train.length + 1, loss: typeof r.loss === 'number' ? r.loss : null };
      if (r.nan) {
        p.nan = true;
      }
      if (typeof r.acc === 'number') {
        p.acc = r.acc;
      }
      if (typeof r.epoch === 'number') {
        p.epoch = r.epoch;
      }
      if (typeof r.lr === 'number') {
        p.lr = r.lr;
      }
      series.train.push(p);
    } else if (r.agg && (r.split === 'train' || r.split === 'val') && typeof r.epoch === 'number') {
      let e = byEpoch.get(r.epoch);
      if (!e) {
        e = { epoch: r.epoch };
        byEpoch.set(r.epoch, e);
      }
      const prefix = r.split === 'val' ? 'val_' : '';
      if (typeof r.loss === 'number') {
        (e as unknown as Record<string, number>)[prefix + 'loss'] = r.loss;
      } else if (r.loss === null) {
        series.valNan = series.valNan || r.split === 'val';
      }
      if (typeof r.acc === 'number') {
        (e as unknown as Record<string, number>)[prefix + 'acc'] = r.acc;
      }
    } else if (r.split === 'val' && r.nan) {
      series.valNan = true;
    }
  }
  series.epochs = [...byEpoch.values()].sort((a, b) => a.epoch - b.epoch);
  return series;
}
