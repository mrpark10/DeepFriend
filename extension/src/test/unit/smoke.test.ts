import { describe, expect, it } from 'vitest';
import { EXTENSION_ID } from '../../constants';

describe('smoke', () => {
  it('has the marketplace id', () => {
    expect(EXTENSION_ID).toBe('mrpark10.deepfriend');
  });
});
