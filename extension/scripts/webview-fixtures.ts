// Builds realistic webview messages from fixture runs, for screenshot QA (scripts/render-webviews.mjs).
import * as fs from 'node:fs';
import * as path from 'node:path';
import { buildChart } from '../src/f2/chart';
import { diagnose } from '../src/f2/doctorRules';
import { actionLabel, renderFinding } from '../src/f2/doctorText';
import { parseMetrics } from '../src/f2/runReader';
import { setLanguage, t } from '../src/i18n/i18n';

const FIX = path.join(__dirname, '..', 'src', 'test', 'fixtures', 'runs');
const out: Record<string, unknown> = {};
for (const lang of ['ko', 'en'] as const) {
  setLanguage(lang);
  for (const name of ['overfit', 'healthy', 'nan', 'chance']) {
    const meta = JSON.parse(fs.readFileSync(path.join(FIX, name, 'meta.json'), 'utf8'));
    const series = parseMetrics(fs.readFileSync(path.join(FIX, name, 'metrics.jsonl'), 'utf8'));
    series.numClasses = meta.numClasses;
    const findings = diagnose(series);
    out[`doctor-${name}-${lang}`] = {
      type: 'state',
      L: { what: t('doctor.what'), why: t('doctor.why'), try: t('doctor.try'), loss: t('doctor.loss'), acc: t('doctor.acc'), train: t('doctor.train'), val: t('doctor.val'), step: t('doctor.step'), copy: t('doctor.empty.copy'), copied: t('doctor.empty.copied'), emptyTitle: t('doctor.empty.title'), emptyBody: t('doctor.empty.body'), ai: t('ai.badge') },
      run: { title: meta.name, status: t(`doctor.status.${meta.status}` as never), meta: t('doctor.meta', { epoch: series.epochs.length, step: series.train.length, lr: String(meta.config.lr), batch: String(meta.config.batch_size) }), dir: '' },
      chart: buildChart(series),
      findings: findings.map((f) => ({ id: f.id, ...renderFinding(f), ai: false, actions: f.actions.map(actionLabel) })),
      note: t('doctor.allGood'),
    };
  }
  out[`doctor-empty-${lang}`] = { ...(out[`doctor-healthy-${lang}`] as object), run: null, findings: [] };
  out[`samples-${lang}`] = JSON.parse(fs.readFileSync(path.join(FIX, 'chance', 'samples.json'), 'utf8'));
}
console.log(JSON.stringify(out));
