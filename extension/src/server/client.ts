import type { Readable, Writable } from 'node:stream';
import * as path from 'node:path';
import type { PythonInfo } from '../python/findPython';
import { LineSplitter } from './protocol';
import type { InitializeResult } from './types';

export type Status = 'starting' | 'ready' | 'busy' | 'error' | 'noPython';

export interface ProcessLike {
  stdin: Writable;
  stdout: Readable;
  stderr: Readable;
  kill(): void;
  on(event: 'exit', cb: (code: number | null, signal: string | null) => void): unknown;
  on(event: 'error', cb: (err: Error) => void): unknown;
}

export type Spawner = (cmd: string, args: string[], env: NodeJS.ProcessEnv) => ProcessLike;

type Listener<T> = (v: T) => void;

/** Minimal event emitter so this module stays free of the `vscode` import (unit-testable). */
export class Emitter<T> {
  private listeners = new Set<Listener<T>>();
  event = (l: Listener<T>): { dispose(): void } => {
    this.listeners.add(l);
    return { dispose: () => this.listeners.delete(l) };
  };
  fire(v: T): void {
    for (const l of [...this.listeners]) {
      l(v);
    }
  }
}

interface Pending {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

const MAX_CRASHES = 5;
const CRASH_WINDOW_MS = 180_000;
const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * Owns the Python analysis server process: start, request/response, and
 * automatic restart with exponential backoff. A crashing server never
 * affects VS Code; after too many crashes we stop and report `error`.
 */
export class AnalysisClient {
  private proc: ProcessLike | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private crashes: number[] = [];
  private disposed = false;
  private restartTimer: ReturnType<typeof setTimeout> | null = null;
  private python: PythonInfo | null = null;
  private bundled = '';
  private statusEmitter = new Emitter<Status>();
  private readyEmitter = new Emitter<InitializeResult>();
  private status_: Status = 'starting';
  info: InitializeResult | null = null;

  readonly onStatus = this.statusEmitter.event;
  readonly onReady = this.readyEmitter.event;

  constructor(private spawner: Spawner, private log: (line: string) => void, private now: () => number = () => Date.now()) {}

  get status(): Status {
    return this.status_;
  }

  private setStatus(s: Status): void {
    this.status_ = s;
    this.statusEmitter.fire(s);
  }

  async start(python: PythonInfo, bundledPath: string): Promise<InitializeResult> {
    this.python = python;
    this.bundled = bundledPath;
    this.crashes = [];
    return this.spawn();
  }

  private spawn(): Promise<InitializeResult> {
    if (!this.python) {
      return Promise.reject(new Error('no python'));
    }
    this.setStatus('starting');
    const env: NodeJS.ProcessEnv = { ...process.env };
    env.PYTHONPATH = this.bundled + (env.PYTHONPATH ? path.delimiter + env.PYTHONPATH : '');
    env.PYTHONIOENCODING = 'utf-8';
    env.PYTHONUNBUFFERED = '1';
    const [cmd, ...pre] = this.python.command;
    let proc: ProcessLike;
    try {
      proc = this.spawner(cmd, [...pre, '-u', '-m', 'deepfriend.server'], env);
    } catch (e) {
      this.setStatus('error');
      return Promise.reject(e as Error);
    }
    this.proc = proc;
    const splitter = new LineSplitter();
    proc.stdout.on('data', (chunk: Buffer) => {
      for (const line of splitter.push(chunk)) {
        this.onLine(line);
      }
    });
    const errSplitter = new LineSplitter();
    proc.stderr.on('data', (chunk: Buffer) => {
      for (const line of errSplitter.push(chunk)) {
        this.log(line);
      }
    });
    proc.on('error', (err: Error) => {
      this.log('spawn error: ' + err.message);
    });
    proc.on('exit', (code, signal) => this.onExit(proc, code, signal));
    return this.request<InitializeResult>('initialize', {}, 60_000).then((info) => {
      this.info = info;
      this.setStatus('ready');
      this.readyEmitter.fire(info);
      return info;
    });
  }

  private onLine(line: string): void {
    let msg: { id?: number; result?: unknown; error?: { message: string }; cancelled?: boolean };
    try {
      msg = JSON.parse(line);
    } catch {
      this.log('bad message: ' + line.slice(0, 200));
      return;
    }
    if (typeof msg.id !== 'number') {
      return;
    }
    const p = this.pending.get(msg.id);
    if (!p) {
      return;
    }
    this.pending.delete(msg.id);
    clearTimeout(p.timer);
    if (msg.error) {
      p.reject(new Error(msg.error.message));
    } else {
      p.resolve(msg.cancelled ? null : msg.result ?? null);
    }
  }

  private onExit(proc: ProcessLike, code: number | null, signal: string | null): void {
    if (proc !== this.proc) {
      return;
    }
    this.proc = null;
    for (const [, p] of this.pending) {
      clearTimeout(p.timer);
      p.reject(new Error('analysis server stopped'));
    }
    this.pending.clear();
    if (this.disposed) {
      return;
    }
    this.log(`analysis server exited (code=${code}, signal=${signal})`);
    const t = this.now();
    this.crashes = this.crashes.filter((c) => t - c < CRASH_WINDOW_MS);
    this.crashes.push(t);
    if (this.crashes.length > MAX_CRASHES) {
      this.setStatus('error');
      return;
    }
    const delay = Math.min(8000, 500 * 2 ** (this.crashes.length - 1));
    this.restartTimer = setTimeout(() => {
      this.restartTimer = null;
      this.spawn().catch((e) => this.log('restart failed: ' + (e as Error).message));
    }, delay);
  }

  request<T = unknown>(method: string, params: unknown, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
    const proc = this.proc;
    if (!proc) {
      return Promise.reject(new Error('analysis server is not running'));
    }
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`request ${method} timed out`));
      }, timeoutMs);
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, timer });
      proc.stdin.write(JSON.stringify({ id, method, params }) + '\n');
    });
  }

  /** Use another interpreter (setting or Python extension changed). */
  switchPython(python: PythonInfo): Promise<InitializeResult> {
    this.python = python;
    return this.restart();
  }

  /** Manual restart (command palette): also resets the crash counter. */
  restart(): Promise<InitializeResult> {
    this.crashes = [];
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
    const old = this.proc;
    this.proc = null;
    if (old) {
      old.kill();
    }
    return this.spawn();
  }

  dispose(): void {
    this.disposed = true;
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
    }
    const proc = this.proc;
    if (proc) {
      try {
        proc.stdin.write(JSON.stringify({ id: this.nextId++, method: 'shutdown', params: {} }) + '\n');
      } catch {
        // ignore
      }
      proc.kill();
    }
  }
}
