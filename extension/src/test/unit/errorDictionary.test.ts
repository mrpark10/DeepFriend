import * as fs from 'node:fs';
import * as path from 'node:path';
import { beforeEach, describe, expect, it } from 'vitest';
import { setLanguage } from '../../i18n/i18n';
import { ENTRIES, explain, renderExplanation } from '../../f3/errorDictionary';
import { findTracebacks } from '../../f3/traceback';

const FIX = path.join(__dirname, '..', 'fixtures', 'errors');
const parse = (name: string) => findTracebacks(fs.readFileSync(path.join(FIX, name), 'utf8'))[0];

const EXPECT: Record<string, [string, Record<string, string>?]> = {
  'shape_matmul.txt': ['shape_matmul', { got: '12544', expected: '10000' }],
  'windows_korean.synthetic.txt': ['shape_matmul', { got: '3136', expected: '10000' }],
  'jupyter.synthetic.txt': ['shape_matmul', { got: '10', expected: '20' }],
  'shape_conv_channels.txt': ['shape_conv_channels', { expected: '3', got: '1' }],
  'shape_tensor_size.txt': ['shape_tensor_size', { a: '3', b: '4', dim: '1' }],
  'shape_mse_warning_free.txt': ['shape_tensor_size'],
  'shape_view.txt': ['shape_view', { shape: '[5, -1]', size: '24' }],
  'shape_target_batch.txt': ['shape_target_batch', { input: '32', target: '64' }],
  'keras_input.txt': ['shape_keras_input', { expected: '(None, 784)', got: '(2, 28, 28)' }],
  'keras_target.txt': ['shape_keras_target', { target: '(None, 1)', output: '(None, 10)' }],
  'cuda_oom.synthetic.txt': ['cuda_oom', { size: '2.00 GiB' }],
  'tf_oom.synthetic.txt': ['tf_oom'],
  'dtype_matmul.txt': ['dtype_matmul', { a: 'Double', b: 'Float' }],
  'dtype_long_target.txt': ['dtype_target', { got: 'Int' }],
  'dtype_float_target.txt': ['dtype_target', { got: 'Float' }],
  'device_mismatch.txt': ['device_mismatch'],
  'device_cuda.synthetic.txt': ['device_mismatch'],
  'device_input_weight.txt': ['device_input_weight'],
  'input_weight_cuda.synthetic.txt': ['device_input_weight'],
  'cuda_unavailable.txt': ['cuda_unavailable'],
  'module_not_found.txt': ['module_not_found', { module: 'cv2', pip: 'opencv-python' }],
  'indentation.txt': ['indentation'],
  'syntax_error.txt': ['syntax_error'],
  'name_error.txt': ['name_error', { name: 'modle' }],
  'attribute_none.txt': ['attribute_none', { attr: 'shape' }],
  'attribute_error.txt': ['attribute_error', { obj: 'Linear', attr: 'fc1' }],
  'type_args.txt': ['type_args'],
  'index_error.txt': ['index_error'],
  'key_error.txt': ['key_error', { key: "'batch_size'" }],
  'file_not_found.txt': ['file_not_found', { path: 'data/train.csv' }],
  'zero_division.txt': ['zero_division'],
};

describe('explain', () => {
  beforeEach(() => setLanguage('ko'));

  for (const [file, [id, vars]] of Object.entries(EXPECT)) {
    it(`${file} → ${id}`, () => {
      const e = parse(file);
      const x = explain(e);
      expect(x?.id).toBe(id);
      if (vars) {
        for (const [k, v] of Object.entries(vars)) {
          expect(x!.vars[k], k).toBe(v);
        }
      }
    });
  }

  it('returns null for unknown errors', () => {
    expect(explain(parse('chained.txt'))).toBeNull();
  });

  it('offers an install button only for missing modules', () => {
    expect(explain(parse('module_not_found.txt'))!.actions).toContain('installModule');
    expect(explain(parse('shape_matmul.txt'))!.actions).not.toContain('installModule');
  });

  it('maps common import names to pip package names', () => {
    const fake = (m: string) => explain({ type: 'ModuleNotFoundError', message: `No module named '${m}'`, frames: [], raw: '' })!.vars.pip;
    expect(fake('sklearn')).toBe('scikit-learn');
    expect(fake('PIL')).toBe('pillow');
    expect(fake('yaml')).toBe('pyyaml');
    expect(fake('torch.nn')).toBe('torch');
    expect(fake('deepfriend')).toBe('deepfriend');
  });

  it('renders text with numbers filled in and no leftover braces, in both languages', () => {
    for (const lang of ['ko', 'en'] as const) {
      setLanguage(lang);
      for (const file of Object.keys(EXPECT)) {
        const x = explain(parse(file))!;
        const r = renderExplanation(x);
        for (const part of [r.title, r.meaning, r.cause, r.fix]) {
          expect(part, `${lang} ${file}`).toBeTruthy();
          expect(part, `${lang} ${file}`).not.toMatch(/\{[a-z]+\}/);
        }
      }
    }
    setLanguage('ko');
    expect(renderExplanation(explain(parse('shape_matmul.txt'))!).meaning).toContain('12544');
  });

  it('every entry has Korean and English for all four parts', () => {
    for (const e of ENTRIES) {
      for (const part of [e.title, e.meaning, e.cause, e.fix]) {
        expect(part.ko, e.id).toBeTruthy();
        expect(part.en, e.id).toBeTruthy();
      }
    }
  });
});
