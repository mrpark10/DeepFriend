import { describe, expect, it, beforeEach } from 'vitest';
import { setLanguage, t, getLanguage } from '../../i18n/i18n';
import { ko } from '../../i18n/ko';
import { en } from '../../i18n/en';

describe('i18n', () => {
  beforeEach(() => setLanguage('ko'));

  it('defaults to Korean', () => {
    expect(getLanguage()).toBe('ko');
    expect(t('status.ready')).toBe(ko['status.ready']);
  });

  it('switches to English', () => {
    setLanguage('en');
    expect(t('status.ready')).toBe(en['status.ready']);
  });

  it('substitutes named arguments, repeatedly', () => {
    expect(t('python.tooOld', { version: '3.8.10', min: '3.9' })).toContain('3.8.10');
    expect(t('python.tooOld', { version: '3.8.10', min: '3.9' })).toContain('3.9');
  });

  it('leaves unknown placeholders visible instead of crashing', () => {
    expect(t('python.tooOld', {})).toContain('{version}');
  });

  it('has a non-empty English string for every Korean key', () => {
    for (const k of Object.keys(ko) as (keyof typeof ko)[]) {
      expect(en[k], k).toBeTruthy();
    }
  });

  it('keeps the same placeholders in both languages', () => {
    const ph = (s: string) => (s.match(/\{[a-zA-Z0-9_]+\}/g) ?? []).sort().join(',');
    for (const k of Object.keys(ko) as (keyof typeof ko)[]) {
      expect(ph(en[k]), k).toBe(ph(ko[k]));
    }
  });

  it('falls back to Korean for unknown language codes', () => {
    setLanguage('fr' as never);
    expect(t('status.ready')).toBe(ko['status.ready']);
  });
});
