import { beforeEach, describe, expect, it } from 'vitest';
import { setLanguage } from '../../i18n/i18n';
import { diagMessage, fixTitle, hintTooltipLines, normalizeInputShape, unknownText } from '../../f1/messages';
import type { Diag } from '../../server/types';

const d = (code: Diag['code'], args: Diag['args']): Diag => ({ line: 0, col: 0, endLine: 0, endCol: 1, code, args });

describe('F1 messages', () => {
  beforeEach(() => setLanguage('ko'));

  it('uses the exact spec sentence for Linear mismatch', () => {
    expect(diagMessage(d('linear_in', { expected: 10000, actual: 12544 }))).toBe(
      '앞 레이어 출력은 12544개인데 Linear는 10000개를 기대해요. in_features를 12544로 바꿀까요?',
    );
  });

  it('translates invalid_arg reasons and tolerates missing args', () => {
    expect(diagMessage(d('invalid_arg', { layer: 'MaxPool2d', reason: 'pool_padding', value: 2 }))).toBe(
      'MaxPool2d 설정값이 잘못됐어요: padding은 kernel_size의 절반 이하여야 해요',
    );
    expect(diagMessage(d('invalid_arg', { reason: 'dim_range', dim: 3, rank: 2 }))).toContain('축 번호');
    expect(diagMessage(d('invalid_arg', { reason: 'something_new' }))).toContain('값을 다시 확인');
  });

  it('fills missing placeholders with ? instead of leaking {braces}', () => {
    expect(diagMessage(d('rank', { layer: 'Conv2d' }))).not.toMatch(/\{|\}/);
  });

  it('switches to English', () => {
    setLanguage('en');
    expect(diagMessage(d('linear_in', { expected: 10, actual: 12 }))).toBe('The previous layer outputs 12 values but Linear expects 10. Change in_features to 12?');
    expect(fixTitle(d('conv_in', { expected: 3, actual: 1 }))).toBe('Change in_channels to 1');
  });

  it('builds fix titles', () => {
    expect(fixTitle(d('linear_in', { expected: 10000, actual: 3136 }))).toBe('in_features를 3136로 바꾸기');
    expect(fixTitle(d('cat_mismatch', {}))).toBeNull();
  });

  it('explains unknown shapes', () => {
    expect(unknownText('ctor_arg')).toContain('기본값');
    expect(unknownText('unsupported_op:F.interpolate')).toContain('F.interpolate');
    expect(unknownText('weird')).toBe(unknownText('unknown'));
  });

  it('lists all shapes when a layer is called several times', () => {
    expect(hintTooltipLines({ line: 1, shape: '(1, 6, 14, 14)', all: ['(1, 6, 14, 14)', '(1, 16, 5, 5)'] }, false)).toEqual([
      '여러 번 호출돼서 크기가 여러 개예요:',
      '- (1, 6, 14, 14)',
      '- (1, 16, 5, 5)',
      '코드를 읽어서 계산한 크기예요.',
    ]);
  });

  it('normalizes and validates input shapes', () => {
    expect(normalizeInputShape('(32,3,224,224)')).toBe('(32, 3, 224, 224)');
    expect(normalizeInputShape(' 32, 3, 224, 224 ')).toBe('(32, 3, 224, 224)');
    expect(normalizeInputShape('(B, 1, 28, 28)')).toBe('(B, 1, 28, 28)');
    expect(normalizeInputShape('(None, 28, 28, 1)')).toBe('(None, 28, 28, 1)');
    expect(normalizeInputShape('(3, 5), (3, 7)')).toBe('(3, 5), (3, 7)');
    expect(normalizeInputShape('(24,)')).toBe('(24,)');
    expect(normalizeInputShape('(0, 3)')).toBeNull();
    expect(normalizeInputShape('(3.5, 2)')).toBeNull();
    expect(normalizeInputShape('hello')).toBeNull();
    expect(normalizeInputShape('()')).toBeNull();
  });
});
