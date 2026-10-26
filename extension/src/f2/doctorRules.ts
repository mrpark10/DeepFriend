// F2 training doctor: deterministic rules. Thresholds were calibrated on examples/05_doctor
// (see test/fixtures/runs); AI may only rephrase the resulting text, never decide.

export interface TrainPoint {
  step: number;
  loss: number | null;
  nan?: boolean;
  acc?: number;
  epoch?: number;
  lr?: number;
}

export interface EpochPoint {
  epoch: number;
  loss?: number;
  acc?: number;
  val_loss?: number;
  val_acc?: number;
}

export interface RunSeries {
  train: TrainPoint[];
  epochs: EpochPoint[];
  numClasses: number | null;
  valNan?: boolean;
}

export type FindingId = 'nan' | 'plateau' | 'chance' | 'overfit' | 'oscillation';

export type DoctorAction =
  | { kind: 'setLr'; factor: number }
  | { kind: 'setBatch'; factor: number }
  | { kind: 'snippet'; id: 'earlyStopping' | 'dropout' | 'normalize' | 'eps' }
  | { kind: 'showSamples' };

export interface Finding {
  id: FindingId;
  args: Record<string, number>;
  actions: DoctorAction[];
}

/** Steps needed before step-based rules judge anything. */
const MIN_STEPS = 50;
/** Relative change below this is "barely moving" (plateau). */
const FLAT = 0.02;
/** A converged run (loss below 10% of its starting value) is never a plateau. */
const CONVERGED = 0.1;
/** Overfitting: validation loss this much above its best, while training loss keeps falling. */
const OVERFIT_RISE = 0.1;
/** Oscillation: residual noise relative to the starting loss, with less than 10% progress. */
const OSC_NOISE = 0.3;
const OSC_PROGRESS = 0.1;
/** Noise this large is oscillation even if the loss happens to trend down. */
const OSC_SEVERE = 0.6;
/** Chance level: best accuracy ≤ 1.5× chance and last ≤ 1.2× chance. */
const CHANCE_BEST = 1.5;
const CHANCE_LAST = 1.2;

const finite = (v: number | null | undefined): v is number => typeof v === 'number' && Number.isFinite(v);
const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
const round = (v: number, d = 1): number => Math.round(v * 10 ** d) / 10 ** d;

/** Standard deviation of the residuals after a least-squares line fit. */
function residualStd(ys: number[]): number {
  const n = ys.length;
  const mx = (n - 1) / 2;
  const my = mean(ys);
  let sxy = 0;
  let sxx = 0;
  ys.forEach((y, i) => {
    sxy += (i - mx) * (y - my);
    sxx += (i - mx) ** 2;
  });
  const b = sxx ? sxy / sxx : 0;
  const res = ys.map((y, i) => y - (my + b * (i - mx)));
  return Math.sqrt(mean(res.map((r) => r * r)));
}

function startScale(s: RunSeries, losses: number[]): number | null {
  if (losses.length) {
    return median(losses.slice(0, 5));
  }
  const e = s.epochs.find((x) => finite(x.loss));
  return e ? (e.loss as number) : null;
}

export function diagnose(s: RunSeries): Finding[] {
  const losses = s.train.filter((p) => finite(p.loss)).map((p) => p.loss as number);
  // 1) NaN / Inf
  const nanStep = s.train.find((p) => p.nan || (p.loss !== null && !Number.isFinite(p.loss)));
  if (nanStep || s.valNan) {
    return [{ id: 'nan', args: { step: nanStep?.step ?? 0 }, actions: [{ kind: 'setLr', factor: 0.1 }, { kind: 'snippet', id: 'eps' }] }];
  }
  const found: Finding[] = [];
  const scale = startScale(s, losses);
  const window = losses.slice(-200);
  const noise = window.length >= 30 && scale ? residualStd(window) / Math.abs(scale) : 0;

  // 2) plateau
  const val = s.epochs.filter((e) => finite(e.val_loss));
  let plateau: number | null = null;
  if (val.length >= 3) {
    const a = val[0].val_loss as number;
    const b = val[val.length - 1].val_loss as number;
    plateau = Math.abs(a - b) / Math.max(Math.abs(a), 1e-12);
    if (scale && b < CONVERGED * scale) {
      plateau = null;
    }
  } else if (losses.length >= MIN_STEPS && noise < OSC_NOISE) {
    const k = Math.max(5, Math.floor(losses.length / 10));
    const a = mean(losses.slice(0, k));
    const b = mean(losses.slice(-k));
    plateau = Math.abs(a - b) / Math.max(Math.abs(a), 1e-12);
    if (scale && b < CONVERGED * scale) {
      plateau = null;
    }
  }
  if (plateau !== null && plateau < FLAT) {
    found.push({ id: 'plateau', args: { changePct: round(plateau * 100) }, actions: [{ kind: 'setLr', factor: 10 }, { kind: 'snippet', id: 'normalize' }] });
  }

  // 3) chance-level accuracy
  const C = s.numClasses;
  if (C && C >= 2) {
    let accs = s.epochs.filter((e) => finite(e.val_acc)).map((e) => e.val_acc as number);
    if (accs.length < 3) {
      accs = s.epochs.filter((e) => finite(e.acc)).map((e) => e.acc as number);
    }
    const enough = s.epochs.length >= 3 || s.train.length >= 200;
    if (accs.length < 3) {
      const stepAcc = s.train.filter((p) => finite(p.acc)).map((p) => p.acc as number);
      accs = [];
      for (let i = 0; i + 50 <= stepAcc.length; i += 50) {
        accs.push(mean(stepAcc.slice(i, i + 50)));
      }
    }
    if (enough && accs.length >= 2) {
      const chance = 1 / C;
      const best = Math.max(...accs);
      const last = accs[accs.length - 1];
      if (best <= CHANCE_BEST * chance && last <= CHANCE_LAST * chance) {
        found.push({ id: 'chance', args: { accPct: Math.round(last * 100), chancePct: Math.round(chance * 100), classes: C }, actions: [{ kind: 'showSamples' }] });
      }
    }
  }

  // 4) overfitting
  if (val.length >= 3) {
    const vl = val.map((e) => e.val_loss as number);
    const b = vl.indexOf(Math.min(...vl));
    const L = vl.length - 1;
    if (b <= L - 2) {
      const tail = vl.slice(Math.max(b + 1, L - 2));
      const rise = mean(tail) / vl[b] - 1;
      const trB = val[b].loss;
      const trL = val[L].loss;
      const trainFalling = finite(trB) && finite(trL) ? trL < trB * 0.9 : true;
      if (rise >= OVERFIT_RISE && trainFalling) {
        found.push({
          id: 'overfit',
          args: { bestEpoch: val[b].epoch + 1, lastEpoch: val[L].epoch + 1, risePct: Math.round(rise * 100) },
          actions: [{ kind: 'snippet', id: 'earlyStopping' }, { kind: 'snippet', id: 'dropout' }],
        });
      }
    }
  }

  // 5) oscillation
  if (window.length >= 30 && scale) {
    const q = Math.max(1, Math.floor(window.length / 4));
    const first = mean(window.slice(0, q));
    const progress = (first - mean(window.slice(-q))) / Math.max(Math.abs(first), 1e-12);
    if ((noise >= OSC_NOISE && progress < OSC_PROGRESS) || noise >= OSC_SEVERE) {
      found.push({ id: 'oscillation', args: { noise: round(noise, 2) }, actions: [{ kind: 'setLr', factor: 0.1 }, { kind: 'setBatch', factor: 2 }] });
    }
  }
  return found;
}
