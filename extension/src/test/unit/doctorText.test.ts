import { beforeEach, describe, expect, it } from 'vitest';
import { setLanguage } from '../../i18n/i18n';
import { actionLabel, editForAction, formatNumber, renderFinding, snippet } from '../../f2/doctorText';
import type { ArgLocation } from '../../server/types';

describe('doctor text', () => {
  beforeEach(() => setLanguage('ko'));

  it('renders the three-line card with numbers', () => {
    const r = renderFinding({ id: 'overfit', args: { bestEpoch: 3, lastEpoch: 10, risePct: 42 }, actions: [] });
    expect(r.what).toContain('3번째 에폭');
    expect(r.what).toContain('42%');
    expect(r.why).toContain('과적합');
    expect(r.try).toContain('Dropout');
    const c = renderFinding({ id: 'chance', args: { accPct: 10, chancePct: 10, classes: 10 }, actions: [] });
    expect(c.what).toContain('10%');
  });

  it('labels actions', () => {
    expect(actionLabel({ kind: 'setLr', factor: 0.1 })).toBe('학습률 1/10로 바꾸기');
    expect(actionLabel({ kind: 'setLr', factor: 10 })).toBe('학습률 10배로 바꾸기');
    expect(actionLabel({ kind: 'setBatch', factor: 2 })).toBe('배치 크기 2배로 바꾸기');
    expect(actionLabel({ kind: 'showSamples' })).toBe('데이터·라벨 샘플 보기');
    setLanguage('en');
    expect(actionLabel({ kind: 'setLr', factor: 0.1 })).toBe('Divide learning rate by 10');
  });

  it('formats numbers the way people write them in code', () => {
    expect(formatNumber(0.1 * 0.1)).toBe('0.01');
    expect(formatNumber(0.03 * 0.1)).toBe('0.003');
    expect(formatNumber(0.001 * 0.1)).toBe('1e-4');
    expect(formatNumber(2.5e-5)).toBe('2.5e-5');
    expect(formatNumber(10 * 0.1)).toBe('1.0');
    expect(formatNumber(500 * 0.1)).toBe('50.0');
  });

  it('builds the replacement for an existing lr literal', () => {
    const loc: ArgLocation = { line: 3, col: 40, endCol: 44, value: 0.01, insert: false, name: 'lr' };
    expect(editForAction(loc, { kind: 'setLr', factor: 0.1 })).toEqual({ line: 3, col: 40, endCol: 44, text: '0.001' });
  });

  it('inserts a keyword argument when the argument is missing', () => {
    const loc: ArgLocation = { line: 1, col: 35, endCol: 35, value: null, insert: true, name: 'lr', hasArgs: true, defaultValue: 0.001 };
    expect(editForAction(loc, { kind: 'setLr', factor: 10 })).toEqual({ line: 1, col: 35, endCol: 35, text: ', lr=0.01' });
    const k: ArgLocation = { line: 1, col: 30, endCol: 30, value: null, insert: true, name: 'learning_rate', hasArgs: false, defaultValue: 0.001 };
    expect(editForAction(k, { kind: 'setLr', factor: 0.1 })!.text).toBe('learning_rate=1e-4');
  });

  it('doubles the batch size as an integer and refuses unknown values', () => {
    const loc: ArgLocation = { line: 2, col: 30, endCol: 31, value: 2, insert: false, name: 'batch_size' };
    expect(editForAction(loc, { kind: 'setBatch', factor: 2 })!.text).toBe('4');
    expect(editForAction({ ...loc, value: null }, { kind: 'setBatch', factor: 2 })).toBeNull();
  });

  it('has framework-specific snippets', () => {
    expect(snippet('earlyStopping', 'pytorch')).toContain('best_val');
    expect(snippet('earlyStopping', 'tensorflow')).toContain('EarlyStopping');
    expect(snippet('dropout', 'pytorch')).toContain('nn.Dropout');
    expect(snippet('normalize', 'pytorch')).toContain('Normalize');
    expect(snippet('eps', 'pytorch')).toContain('1e-8');
  });
});

describe('snippets in English', () => {
  it('contain no Korean when the language is English', () => {
    setLanguage('en');
    for (const id of ['earlyStopping', 'dropout', 'normalize', 'eps']) {
      for (const fw of ['pytorch', 'tensorflow'] as const) {
        expect(/[가-힣]/.test(snippet(id, fw)), `${id} ${fw}`).toBe(false);
      }
    }
    setLanguage('ko');
  });
});
