import { defineConfig } from '@vscode/test-cli';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const workspace = join(here, 'src', 'test', 'fixtures', 'workspace');
const python = process.env.DEEPFRIEND_TEST_PYTHON || join(here, '..', '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');

// Start every run from a clean slate: no runs left over from a previous test session.
rmSync(join(workspace, '.deepfriend'), { recursive: true, force: true });

// Generated per run so the absolute interpreter path never gets committed.
mkdirSync(join(workspace, '.vscode'), { recursive: true });
writeFileSync(join(workspace, '.vscode', 'settings.json'), JSON.stringify({ 'deepfriend.pythonPath': python }, null, 2));

export default defineConfig({
  files: 'out/test/suite/**/*.test.js',
  version: process.env.DEEPFRIEND_TEST_VSCODE || '1.93.0',
  workspaceFolder: workspace,
  // DEEPFRIEND_TEST_EXT_PATH lets the system test run the suite against an unpacked VSIX.
  extensionDevelopmentPath: [process.env.DEEPFRIEND_TEST_EXT_PATH || here, join(here, 'src', 'test', 'fixtures', 'nb-ext')],
  launchArgs: ['--disable-extensions', '--disable-workspace-trust'],
  env: { DEEPFRIEND_TEST_PYTHON: python },
  mocha: { ui: 'tdd', timeout: 60000 },
});
