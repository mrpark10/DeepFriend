import type { RunSeries } from './doctorRules';

export interface ChartData {
  x: number[];
  trainLoss: (number | null)[];
  valLoss: (number | null)[];
  trainAcc: (number | null)[];
  valAcc: (number | null)[];
  hasAcc: boolean;
}

const MAX_POINTS = 2000;

/** Series for the doctor charts: one x axis (steps), validation values at each epoch's last step. */
export function buildChart(s: RunSeries): ChartData {
  if (!s.train.length) {
    const ep = s.epochs;
    return {
      x: ep.map((e) => e.epoch + 1),
      trainLoss: ep.map((e) => e.loss ?? null),
      valLoss: ep.map((e) => e.val_loss ?? null),
      trainAcc: ep.map((e) => e.acc ?? null),
      valAcc: ep.map((e) => e.val_acc ?? null),
      hasAcc: ep.some((e) => e.acc !== undefined || e.val_acc !== undefined),
    };
  }
  const lastStepOfEpoch = new Map<number, number>();
  for (const p of s.train) {
    if (p.epoch !== undefined) {
      lastStepOfEpoch.set(p.epoch, Math.max(lastStepOfEpoch.get(p.epoch) ?? 0, p.step));
    }
  }
  const bucket = Math.max(1, Math.ceil(s.train.length / MAX_POINTS));
  const x: number[] = [];
  const trainLoss: (number | null)[] = [];
  const trainAcc: (number | null)[] = [];
  for (let i = 0; i < s.train.length; i += bucket) {
    const chunk = s.train.slice(i, i + bucket);
    const losses = chunk.map((p) => p.loss).filter((v): v is number => v !== null && Number.isFinite(v));
    const accs = chunk.map((p) => p.acc).filter((v): v is number => typeof v === 'number');
    x.push(chunk[chunk.length - 1].step);
    trainLoss.push(losses.length ? losses.reduce((a, b) => a + b, 0) / losses.length : null);
    trainAcc.push(accs.length ? accs.reduce((a, b) => a + b, 0) / accs.length : null);
  }
  const valLoss: (number | null)[] = x.map(() => null);
  const valAcc: (number | null)[] = x.map(() => null);
  for (const e of s.epochs) {
    if (e.val_loss === undefined && e.val_acc === undefined) {
      continue;
    }
    const step = lastStepOfEpoch.get(e.epoch) ?? x[x.length - 1];
    let idx = x.findIndex((v) => v >= step);
    if (idx < 0) {
      idx = x.length - 1;
    }
    if (e.val_loss !== undefined) {
      valLoss[idx] = e.val_loss;
    }
    if (e.val_acc !== undefined) {
      valAcc[idx] = e.val_acc;
    }
  }
  const hasAcc = trainAcc.some((v) => v !== null) || valAcc.some((v) => v !== null);
  return { x, trainLoss, valLoss, trainAcc, valAcc, hasAcc };
}
