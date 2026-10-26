// Copies the Python package, examples and vendored webview libraries into the
// extension folder so the VSIX is self-contained (no pip install needed).
import { cpSync, rmSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const ext = join(here, '..');
const root = join(ext, '..');

const skip = new Set(['tests', '__pycache__', '.pytest_cache']);
const filter = (src) => !skip.has(basename(src)) && !src.endsWith('.pyc');

rmSync(join(ext, 'python'), { recursive: true, force: true });
cpSync(join(root, 'python', 'deepfriend'), join(ext, 'python', 'deepfriend'), { recursive: true, filter });

rmSync(join(ext, 'examples'), { recursive: true, force: true });
if (existsSync(join(root, 'examples'))) {
  cpSync(join(root, 'examples'), join(ext, 'examples'), { recursive: true, filter });
}

const vendor = join(ext, 'media', 'vendor');
mkdirSync(vendor, { recursive: true });
cpSync(join(ext, 'node_modules', 'uplot', 'dist', 'uPlot.iife.min.js'), join(vendor, 'uPlot.iife.min.js'));
cpSync(join(ext, 'node_modules', 'uplot', 'dist', 'uPlot.min.css'), join(vendor, 'uPlot.min.css'));

// Marketplace README/CHANGELOG: copied from the repository root with absolute image URLs.
import { readFileSync, writeFileSync } from 'node:fs';
const RAW = 'https://github.com/mrpark10/DeepFriend/raw/HEAD/';
const BLOB = 'https://github.com/mrpark10/DeepFriend/blob/HEAD/';
const readme = readFileSync(join(root, 'README.md'), 'utf8')
  .replace(/\]\((docs\/images\/[^)]+)\)/g, (_m, p) => `](${RAW}${p})`)
  .replace(/\]\(((?:docs\/|README\.en)[^)]+)\)/g, (_m, p) => `](${BLOB}${p})`);
writeFileSync(join(ext, 'README.md'), readme);
cpSync(join(root, 'CHANGELOG.md'), join(ext, 'CHANGELOG.md'));
