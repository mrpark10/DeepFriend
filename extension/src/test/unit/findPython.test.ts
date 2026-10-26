import { describe, expect, it } from 'vitest';
import { findPython, FindDeps, MIN_PYTHON } from '../../python/findPython';

type Probe = Record<string, { version: string; executable: string } | null>;

function deps(platform: NodeJS.Platform, probes: Probe, onPath: string[] = []): FindDeps & { probed: string[] } {
  const probed: string[] = [];
  return {
    platform,
    probed,
    which: async (cmd) => (onPath.includes(cmd) ? '/usr/bin/' + cmd : null),
    probe: async (cmd) => {
      const key = cmd.join(' ');
      probed.push(key);
      return probes[key] ?? null;
    },
  };
}

describe('findPython', () => {
  it('uses the setting first', async () => {
    const d = deps('darwin', { '/my/py': { version: '3.11.2', executable: '/my/py' }, python3: { version: '3.12.0', executable: '/usr/bin/python3' } }, ['python3']);
    const r = await findPython({ setting: '/my/py', pythonExtPath: '/ext/py' }, d);
    expect(r).toEqual({ command: ['/my/py'], version: '3.11.2', source: 'setting' });
  });

  it('then the Python extension interpreter', async () => {
    const d = deps('linux', { '/ext/py': { version: '3.10.4', executable: '/ext/py' } });
    const r = await findPython({ pythonExtPath: '/ext/py' }, d);
    expect(r?.source).toBe('pythonExtension');
    expect(r?.command).toEqual(['/ext/py']);
  });

  it('falls back to PATH and resolves the real executable', async () => {
    const d = deps('darwin', { python3: { version: '3.9.6', executable: '/usr/bin/python3' } }, ['python3']);
    const r = await findPython({}, d);
    expect(r).toEqual({ command: ['/usr/bin/python3'], version: '3.9.6', source: 'path' });
  });

  it('rejects interpreters older than the minimum and keeps looking', async () => {
    const d = deps('linux', { '/old/py': { version: '3.8.10', executable: '/old/py' }, python3: { version: '3.12.1', executable: '/usr/bin/python3' } }, ['python3']);
    const r = await findPython({ setting: '/old/py' }, d);
    expect(r?.version).toBe('3.12.1');
    expect(MIN_PYTHON).toEqual([3, 9]);
  });

  it('reports the best too-old interpreter when nothing fits', async () => {
    const d = deps('linux', { python3: { version: '3.8.10', executable: '/usr/bin/python3' } }, ['python3']);
    const r = await findPython({}, d);
    expect(r).toBeNull();
  });

  it('tries the py launcher first on Windows, with Korean paths', async () => {
    const exe = 'C:\\Users\\박준호\\AppData\\Local\\Programs\\Python\\Python312\\python.exe';
    const d = deps('win32', { 'py -3': { version: '3.12.3', executable: exe } }, ['py', 'python']);
    const r = await findPython({}, d);
    expect(r).toEqual({ command: [exe], version: '3.12.3', source: 'path' });
    expect(d.probed[0]).toBe('py -3');
  });

  it('skips the Windows Store stub (probe returns nothing)', async () => {
    const d = deps('win32', { python: null, python3: { version: '3.11.0', executable: 'C:\\Python311\\python.exe' } }, ['python', 'python3']);
    const r = await findPython({}, d);
    expect(r?.command).toEqual(['C:\\Python311\\python.exe']);
  });

  it('ignores an empty setting', async () => {
    const d = deps('darwin', { python3: { version: '3.12.0', executable: '/usr/bin/python3' } }, ['python3']);
    const r = await findPython({ setting: '   ' }, d);
    expect(r?.source).toBe('path');
  });
});
