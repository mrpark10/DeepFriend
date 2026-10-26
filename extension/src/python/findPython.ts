export const MIN_PYTHON: [number, number] = [3, 9];

export interface PythonInfo {
  command: string[];
  version: string;
  source: 'setting' | 'pythonExtension' | 'path';
}

export interface FindDeps {
  platform: NodeJS.Platform;
  which: (cmd: string) => Promise<string | null>;
  /** Run `cmd -c <script>`; resolves the interpreter version and real executable, or null. */
  probe: (cmd: string[]) => Promise<{ version: string; executable: string } | null>;
}

export function versionOk(version: string): boolean {
  const [maj, min] = version.split('.').map((x) => parseInt(x, 10));
  if (Number.isNaN(maj) || Number.isNaN(min)) {
    return false;
  }
  return maj > MIN_PYTHON[0] || (maj === MIN_PYTHON[0] && min >= MIN_PYTHON[1]);
}

/**
 * Find a usable Python: the user's setting, then the interpreter selected in
 * the Microsoft Python extension, then PATH (the `py` launcher first on Windows).
 */
export async function findPython(opts: { setting?: string; pythonExtPath?: string }, deps: FindDeps): Promise<PythonInfo | null> {
  const candidates: { cmd: string[]; source: PythonInfo['source']; needsWhich: boolean }[] = [];
  if (opts.setting && opts.setting.trim()) {
    candidates.push({ cmd: [opts.setting.trim()], source: 'setting', needsWhich: false });
  }
  if (opts.pythonExtPath && opts.pythonExtPath.trim()) {
    candidates.push({ cmd: [opts.pythonExtPath.trim()], source: 'pythonExtension', needsWhich: false });
  }
  const pathCmds = deps.platform === 'win32' ? [['py', '-3'], ['python'], ['python3']] : [['python3'], ['python']];
  for (const cmd of pathCmds) {
    candidates.push({ cmd, source: 'path', needsWhich: true });
  }
  for (const c of candidates) {
    if (c.needsWhich && !(await deps.which(c.cmd[0]))) {
      continue;
    }
    const info = await deps.probe(c.cmd);
    if (!info || !versionOk(info.version)) {
      continue;
    }
    return { command: [info.executable || c.cmd[0]], version: info.version, source: c.source };
  }
  return null;
}
