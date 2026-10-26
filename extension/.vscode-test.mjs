import { defineConfig } from '@vscode/test-cli';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  files: 'out/test/suite/**/*.test.js',
  version: '1.93.0',
  workspaceFolder: join(here, 'src', 'test', 'fixtures', 'workspace'),
  launchArgs: ['--disable-extensions', '--disable-workspace-trust'],
  env: { DEEPFRIEND_TEST_PYTHON: join(here, '..', '.venv', 'bin', 'python') },
  mocha: { ui: 'tdd', timeout: 60000 },
});
