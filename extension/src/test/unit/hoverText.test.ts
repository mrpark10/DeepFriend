import { beforeEach, describe, expect, it } from 'vitest';
import { setLanguage } from '../../i18n/i18n';
import { CONCEPTS } from '../../f7/concepts';
import { hoverMarkdown } from '../../f7/hoverText';

describe('hoverMarkdown', () => {
  beforeEach(() => setLanguage('ko'));

  it('shows title, beginner text, diagram, params and doc link', () => {
    const md = hoverMarkdown(CONCEPTS['torch.Conv2d'], 'beginner');
    expect(md).toContain('**nn.Conv2d — 2차원 합성곱 층**');
    expect(md).toContain(CONCEPTS['torch.Conv2d'].what.beginner.ko);
    expect(md).toContain('```text');
    expect(md).toContain('- `kernel_size`: ');
    expect(md).toContain('[공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)');
  });

  it('uses the normal level text when asked', () => {
    const md = hoverMarkdown(CONCEPTS['torch.Conv2d'], 'normal');
    expect(md).toContain(CONCEPTS['torch.Conv2d'].what.normal.ko);
    expect(md).not.toContain(CONCEPTS['torch.Conv2d'].what.beginner.ko);
  });

  it('puts the hovered parameter first and bold', () => {
    const md = hoverMarkdown(CONCEPTS['torch.Conv2d'], 'beginner', 'stride');
    expect(md.startsWith('**`stride`** — ' + CONCEPTS['torch.Conv2d'].params.stride.ko)).toBe(true);
  });

  it('says so when a parameter is not documented', () => {
    const md = hoverMarkdown(CONCEPTS['torch.Conv2d'], 'beginner', 'padding_mode');
    expect(md).toContain('**nn.Conv2d');
  });

  it('includes the shape of the line when known', () => {
    expect(hoverMarkdown(CONCEPTS['torch.Linear'], 'beginner', undefined, '(64, 10)')).toContain('이 줄의 데이터 크기: `(64, 10)`');
  });

  it('renders English', () => {
    setLanguage('en');
    const md = hoverMarkdown(CONCEPTS['keras.Dense'], 'beginner');
    expect(md).toContain('**layers.Dense — fully connected layer**');
    expect(md).toContain('[Official docs]');
  });
});
