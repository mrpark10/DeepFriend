import * as assert from 'assert';
import { execFile } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';
import { api, wsPath } from './helpers';

suite('F6 export to app', () => {
  test('a Keras model becomes .tflite + Flutter code, verified', async function () {
    this.timeout(240_000);
    const a = await api();
    await a.ensureServer();
    const dir = wsPath('f6');
    fs.mkdirSync(dir, { recursive: true });
    const model = path.join(dir, 'tiny.keras');
    await promisify(execFile)(process.env.DEEPFRIEND_TEST_PYTHON!, [
      '-c',
      `import os\nos.environ["TF_CPP_MIN_LOG_LEVEL"]="3"\nimport keras\nfrom keras import layers\nm = keras.Sequential([keras.Input(shape=(8,)), layers.Dense(4, activation="relu"), layers.Dense(3)])\nm.save(${JSON.stringify(model)})`,
    ], { timeout: 180_000 });
    const out = path.join(dir, 'export_out');
    const r = await a.exporter.runExport({ framework: 'keras', modelPath: model, outDir: out });
    assert.strictEqual(r.ok, true, JSON.stringify(r));
    assert.deepStrictEqual(r.inputShape, [1, 8]);
    assert.deepStrictEqual(r.outputShape, [1, 3]);
    assert.strictEqual(r.verify?.passed, true);
    assert.ok(fs.existsSync(path.join(out, 'flutter', 'lib', 'deepfriend_model.dart')));
    assert.ok(fs.readFileSync(path.join(out, 'flutter', 'README.md'), 'utf8').includes('Flutter'));
  });

  test('a broken model file is reported, not crashed', async function () {
    this.timeout(120_000);
    const a = await api();
    const bad = wsPath('f6', 'broken.keras');
    fs.writeFileSync(bad, 'nope');
    const r = await a.exporter.runExport({ framework: 'keras', modelPath: bad, outDir: wsPath('f6', 'out2') });
    assert.strictEqual(r.ok, false);
    assert.strictEqual(r.error?.code, 'load_failed');
  });
});
