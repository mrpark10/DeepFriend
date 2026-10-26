import { execFile } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import type { FindDeps } from './findPython';

const PROBE = "import sys,json;print(json.dumps({'v':'%d.%d.%d'%sys.version_info[:3],'e':sys.executable}))";

function which(cmd: string): Promise<string | null> {
  const dirs = (process.env.PATH ?? '').split(path.delimiter).filter(Boolean);
  const exts = process.platform === 'win32' ? (process.env.PATHEXT ?? '.EXE;.CMD;.BAT').split(';') : [''];
  for (const d of dirs) {
    for (const e of exts) {
      const p = path.join(d, cmd + e.toLowerCase());
      const P = path.join(d, cmd + e);
      for (const cand of [p, P]) {
        try {
          if (fs.statSync(cand).isFile()) {
            return Promise.resolve(cand);
          }
        } catch {
          // keep looking
        }
      }
    }
  }
  return Promise.resolve(null);
}

function probe(cmd: string[]): Promise<{ version: string; executable: string } | null> {
  return new Promise((resolve) => {
    execFile(cmd[0], [...cmd.slice(1), '-c', PROBE], { timeout: 8000, windowsHide: true, encoding: 'utf8' }, (err, stdout) => {
      if (err) {
        resolve(null);
        return;
      }
      try {
        const line = stdout.trim().split(/\r?\n/).pop() ?? '';
        const j = JSON.parse(line) as { v: string; e: string };
        resolve({ version: j.v, executable: j.e });
      } catch {
        resolve(null);
      }
    });
  });
}

export const nodeFindDeps: FindDeps = { platform: process.platform, which, probe };
