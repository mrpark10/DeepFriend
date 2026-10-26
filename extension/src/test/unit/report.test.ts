import { describe, expect, it } from 'vitest';
import { buildIssueUrl } from '../../reportUrl';

describe('buildIssueUrl', () => {
  it('builds a prefilled GitHub issue without user code', () => {
    const url = buildIssueUrl({
      title: '[오류] 분석기 멈춤',
      intro: '무엇을 하다가?',
      extVersion: '0.1.0',
      vscodeVersion: '1.95.0',
      os: 'darwin arm64',
      python: '3.12.1',
      torch: '2.4.0',
      tensorflow: null,
      error: 'Error: boom\n    at x (/Users/박준호/secret/train.py:3)',
    });
    const u = new URL(url);
    expect(u.origin + u.pathname).toBe('https://github.com/mrpark10/DeepFriend/issues/new');
    expect(u.searchParams.get('title')).toBe('[오류] 분석기 멈춤');
    const body = u.searchParams.get('body')!;
    expect(body).toContain('0.1.0');
    expect(body).toContain('1.95.0');
    expect(body).toContain('torch 2.4.0');
    expect(body).toContain('Error: boom');
    // home directory is masked so a username never leaks
    expect(body).not.toContain('박준호');
    expect(body).toContain('~');
  });

  it('keeps the URL under 7000 characters', () => {
    const url = buildIssueUrl({ title: 't', intro: 'i', extVersion: '1', vscodeVersion: '1', os: 'x', python: null, torch: null, tensorflow: null, error: 'x'.repeat(50_000) });
    expect(url.length).toBeLessThan(7000);
  });
});
