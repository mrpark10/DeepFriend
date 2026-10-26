import { describe, expect, it, vi } from 'vitest';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { AnalysisClient, ProcessLike, Status } from '../../server/client';

class FakeProc extends EventEmitter implements ProcessLike {
  stdin = new PassThrough();
  stdout = new PassThrough();
  stderr = new PassThrough();
  killed = false;
  received: any[] = [];
  constructor(private autoReply: boolean | 'init' = true) {
    super();
    this.stdin.on('data', (b: Buffer) => {
      for (const line of b.toString().split('\n').filter(Boolean)) {
        const msg = JSON.parse(line);
        this.received.push(msg);
        if (this.autoReply === true || (this.autoReply === 'init' && msg.method === 'initialize')) {
          const result = msg.method === 'initialize' ? { version: '0.1.0', torch: '2.4.0', tensorflow: null, pythonVersion: '3.12.0', python: '/py' } : { echo: msg.params };
          this.stdout.write(JSON.stringify({ id: msg.id, result }) + '\n');
        }
      }
    });
  }
  kill() {
    this.killed = true;
    this.emit('exit', null, 'SIGTERM');
  }
  crash() {
    this.emit('exit', 1, null);
  }
}

const python = { command: ['/py'], version: '3.12.0', source: 'path' as const };

describe('AnalysisClient', () => {
  it('initializes and reports ready', async () => {
    const procs: FakeProc[] = [];
    const statuses: Status[] = [];
    const c = new AnalysisClient(() => { const p = new FakeProc(); procs.push(p); return p; }, () => {});
    c.onStatus((s) => statuses.push(s));
    const info = await c.start(python, '/bundle');
    expect(info.torch).toBe('2.4.0');
    expect(statuses).toEqual(['starting', 'ready']);
    const r = await c.request('echo', { x: 1 });
    expect(r).toEqual({ echo: { x: 1 } });
    c.dispose();
    expect(procs[0].killed).toBe(true);
  });

  it('passes PYTHONPATH and UTF-8 env to the child', async () => {
    let seenEnv: NodeJS.ProcessEnv = {};
    let seenArgs: string[] = [];
    const c = new AnalysisClient((cmd, args, env) => { seenEnv = env; seenArgs = [cmd, ...args]; return new FakeProc(); }, () => {});
    await c.start(python, '/bundle dir/python');
    expect(seenArgs).toEqual(['/py', '-u', '-m', 'deepfriend.server']);
    expect(seenEnv.PYTHONPATH?.startsWith('/bundle dir/python')).toBe(true);
    expect(seenEnv.PYTHONIOENCODING).toBe('utf-8');
    c.dispose();
  });

  it('resolves cancelled responses with null', async () => {
    const p = new FakeProc(false);
    const c = new AnalysisClient(() => p, () => {});
    const started = c.start(python, '/b');
    p.stdout.write(JSON.stringify({ id: 1, result: { version: '0.1.0' } }) + '\n');
    await started;
    const req = c.request('analyze', {});
    p.stdout.write(JSON.stringify({ id: 2, result: null, cancelled: true }) + '\n');
    expect(await req).toBeNull();
    c.dispose();
  });

  it('rejects server errors', async () => {
    const p = new FakeProc(false);
    const c = new AnalysisClient(() => p, () => {});
    const started = c.start(python, '/b');
    p.stdout.write(JSON.stringify({ id: 1, result: {} }) + '\n');
    await started;
    const req = c.request('boom', {});
    p.stdout.write(JSON.stringify({ id: 2, error: { message: 'bad', trace: '' } }) + '\n');
    await expect(req).rejects.toThrow('bad');
    c.dispose();
  });

  it('restarts after a crash with backoff', async () => {
    vi.useFakeTimers();
    const procs: FakeProc[] = [];
    const c = new AnalysisClient(() => { const p = new FakeProc('init'); procs.push(p); return p; }, () => {});
    await c.start(python, '/b');
    const pending = c.request('echo', {}).catch((e) => e);
    procs[0].crash();
    expect(await pending).toBeInstanceOf(Error);
    await vi.advanceTimersByTimeAsync(600);
    expect(procs.length).toBe(2);
    c.dispose();
    vi.useRealTimers();
  });

  it('gives up after 5 crashes within 3 minutes', async () => {
    vi.useFakeTimers();
    const procs: FakeProc[] = [];
    const statuses: Status[] = [];
    const c = new AnalysisClient(() => { const p = new FakeProc(); procs.push(p); return p; }, () => {});
    c.onStatus((s) => statuses.push(s));
    await c.start(python, '/b');
    for (let i = 0; i < 6; i++) {
      procs[procs.length - 1].crash();
      await vi.advanceTimersByTimeAsync(10_000);
    }
    expect(procs.length).toBe(6);
    expect(statuses[statuses.length - 1]).toBe('error');
    c.dispose();
    vi.useRealTimers();
  });

  it('times out a request that never answers', async () => {
    vi.useFakeTimers();
    const p = new FakeProc(false);
    const c = new AnalysisClient(() => p, () => {});
    const started = c.start(python, '/b');
    p.stdout.write(JSON.stringify({ id: 1, result: {} }) + '\n');
    await started;
    const req = c.request('slow', {}, 1000).catch((e) => e);
    await vi.advanceTimersByTimeAsync(1500);
    expect(await req).toBeInstanceOf(Error);
    c.dispose();
    vi.useRealTimers();
  });
});

describe('AnalysisClient.switchPython', () => {
  it('restarts with the new interpreter and keeps auto-restart working', async () => {
    vi.useFakeTimers();
    const cmds: string[] = [];
    const procs: FakeProc[] = [];
    const c = new AnalysisClient((cmd) => { cmds.push(cmd); const p = new FakeProc(); procs.push(p); return p; }, () => {});
    await c.start(python, '/b');
    await c.switchPython({ command: ['/other/py'], version: '3.11.0', source: 'setting' });
    expect(cmds).toEqual(['/py', '/other/py']);
    expect(procs[0].killed).toBe(true);
    procs[1].crash();
    await vi.advanceTimersByTimeAsync(600);
    expect(cmds).toEqual(['/py', '/other/py', '/other/py']);
    c.dispose();
    vi.useRealTimers();
  });
});
