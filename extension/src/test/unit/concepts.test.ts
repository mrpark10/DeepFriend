import { describe, expect, it } from 'vitest';
import { CONCEPTS, REQUIRED_CONCEPTS } from '../../f7/concepts';
import { resolveAt } from '../../f7/resolve';

function at(text: string, marker: string, delta = 1): number {
  return text.indexOf(marker) + delta;
}

describe('concept data', () => {
  it('covers every required layer', () => {
    for (const k of REQUIRED_CONCEPTS) {
      expect(CONCEPTS[k], k).toBeTruthy();
    }
  });

  it('has Korean and English text for both levels and every parameter', () => {
    for (const [k, c] of Object.entries(CONCEPTS)) {
      for (const lang of ['ko', 'en'] as const) {
        expect(c.title[lang], k).toBeTruthy();
        expect(c.what.beginner[lang], k).toBeTruthy();
        expect(c.what.normal[lang], k).toBeTruthy();
        for (const [p, v] of Object.entries(c.params)) {
          expect(v[lang], `${k}.${p}`).toBeTruthy();
        }
      }
      expect(c.doc.startsWith('https://'), k).toBe(true);
      expect(c.what.beginner.ko).not.toBe(c.what.normal.ko);
    }
  });

  it('documents the common parameters of Conv2d', () => {
    expect(Object.keys(CONCEPTS['torch.Conv2d'].params)).toEqual(
      expect.arrayContaining(['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']),
    );
  });
});

describe('resolveAt', () => {
  const TORCH = 'import torch\nimport torch.nn as nn\n';

  it('finds nn.Conv2d under the cursor', () => {
    const text = TORCH + 'self.c = nn.Conv2d(3, 8, 3)\n';
    expect(resolveAt(text, at(text, 'Conv2d'))).toEqual({ key: 'torch.Conv2d' });
    expect(resolveAt(text, at(text, 'nn.Conv2d'))).toEqual({ key: 'torch.Conv2d' });
  });

  it('handles torch.nn.X and bare imported names', () => {
    const a = 'import torch\nx = torch.nn.Linear(3, 4)\n';
    expect(resolveAt(a, at(a, 'Linear'))).toEqual({ key: 'torch.Linear' });
    const b = 'from torch.nn import ReLU\nact = ReLU()\n';
    expect(resolveAt(b, b.lastIndexOf('ReLU') + 1)).toEqual({ key: 'torch.ReLU' });
  });

  it('resolves Keras names by prefix and by imports', () => {
    const a = 'from tensorflow.keras import layers\nlayers.Dense(10)\n';
    expect(resolveAt(a, at(a, 'Dense'))).toEqual({ key: 'keras.Dense' });
    const b = 'import tensorflow as tf\ntf.keras.layers.Flatten()\n';
    expect(resolveAt(b, at(b, 'Flatten'))).toEqual({ key: 'keras.Flatten' });
    const c = 'from keras.layers import Dropout\nDropout(0.5)\n';
    expect(resolveAt(c, c.lastIndexOf('Dropout') + 1)).toEqual({ key: 'keras.Dropout' });
    const d = TORCH + 'nn.Dropout(0.5)\n';
    expect(resolveAt(d, at(d, 'Dropout'))).toEqual({ key: 'torch.Dropout' });
  });

  it('finds a keyword argument inside a multi-line call', () => {
    const text = TORCH + 'self.c = nn.Conv2d(\n    3,\n    8,\n    kernel_size=3,\n    stride=(2, 2),\n)\n';
    expect(resolveAt(text, at(text, 'kernel_size'))).toEqual({ key: 'torch.Conv2d', param: 'kernel_size' });
    expect(resolveAt(text, at(text, 'stride'))).toEqual({ key: 'torch.Conv2d', param: 'stride' });
  });

  it('finds keyword args of nested calls correctly', () => {
    const text = TORCH + 'nn.Sequential(nn.Conv2d(1, 4, kernel_size=3), nn.ReLU(inplace=True))\n';
    expect(resolveAt(text, at(text, 'kernel_size'))).toEqual({ key: 'torch.Conv2d', param: 'kernel_size' });
    expect(resolveAt(text, at(text, 'inplace'))).toEqual({ key: 'torch.ReLU', param: 'inplace' });
  });

  it('resolves optimizers, losses and DataLoader', () => {
    const text = 'import torch\nfrom torch.utils.data import DataLoader\nopt = torch.optim.Adam(m.parameters(), lr=1e-3)\nloss = nn.CrossEntropyLoss()\ndl = DataLoader(ds, batch_size=32)\n';
    expect(resolveAt(text, at(text, 'Adam'))).toEqual({ key: 'torch.optim.Adam' });
    expect(resolveAt(text, at(text, 'lr='))).toEqual({ key: 'torch.optim.Adam', param: 'lr' });
    expect(resolveAt(text, at(text, 'CrossEntropyLoss'))).toEqual({ key: 'torch.CrossEntropyLoss' });
    expect(resolveAt(text, at(text, 'batch_size'))).toEqual({ key: 'torch.DataLoader', param: 'batch_size' });
  });

  it('returns null for unknown names, comments and strings', () => {
    const text = TORCH + 'x = foo(3)  # nn.Conv2d here\ns = "nn.Linear"\n';
    expect(resolveAt(text, at(text, 'foo'))).toBeNull();
    expect(resolveAt(text, at(text, 'Conv2d'))).toBeNull();
    expect(resolveAt(text, at(text, 'Linear'))).toBeNull();
  });

  it('does not treat comparisons as keyword arguments', () => {
    const text = TORCH + 'if stride == 2:\n    pass\n';
    expect(resolveAt(text, at(text, 'stride'))).toBeNull();
  });
});

describe('diagrams', () => {
  it('have both languages and no Korean in the English version', () => {
    for (const [k, c] of Object.entries(CONCEPTS)) {
      if (!c.diagram) continue;
      expect(c.diagram.ko, k).toBeTruthy();
      expect(c.diagram.en, k).toBeTruthy();
      expect(/[가-힣]/.test(c.diagram.en), k).toBe(false);
    }
  });
});
