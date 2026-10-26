import { describe, expect, it, vi } from 'vitest';
import { AiCore, Provider } from '../../ai/core';

function provider(reply: string | null | (() => Promise<string | null>)): Provider & { calls: { system: string; user: string }[] } {
  const calls: { system: string; user: string }[] = [];
  return {
    calls,
    async complete(system, user) {
      calls.push({ system, user });
      return typeof reply === 'function' ? reply() : reply;
    },
  };
}

const err = { type: 'RuntimeError', message: 'weird thing happened in step 12', frames: [], raw: '' };

describe('AiCore', () => {
  it('does nothing when the provider is off', async () => {
    const consent = vi.fn(async () => true);
    const core = new AiCore({ provider: () => null, consent, language: () => 'ko' });
    expect(await core.explainError(err, '')).toBeNull();
    expect(consent).not.toHaveBeenCalled();
    expect(core.available()).toBe(false);
  });

  it('never calls the provider when consent is denied', async () => {
    const p = provider('{"meaning":"a","cause":"b","fix":"c"}');
    const core = new AiCore({ provider: () => p, consent: async () => false, language: () => 'ko' });
    expect(await core.explainError(err, 'x = 1')).toBeNull();
    expect(p.calls).toHaveLength(0);
  });

  it('shows exactly what will be sent in the consent preview', async () => {
    const p = provider('{"meaning":"뜻","cause":"원인","fix":"해결"}');
    let preview = '';
    const core = new AiCore({ provider: () => p, consent: async (s) => ((preview = s), true), language: () => 'ko' });
    const r = await core.explainError(err, 'a\nb');
    expect(r).toEqual({ meaning: '뜻', cause: '원인', fix: '해결' });
    expect(p.calls[0].user).toBe(preview);
    expect(preview).toContain('RuntimeError: weird thing happened in step 12');
  });

  it('sends at most 10 lines of code', async () => {
    const p = provider('{"meaning":"m","cause":"c","fix":"f"}');
    const core = new AiCore({ provider: () => p, consent: async () => true, language: () => 'ko' });
    const code = Array.from({ length: 50 }, (_, i) => `line${i}`).join('\n');
    await core.explainError(err, code);
    const sent = p.calls[0].user;
    expect(sent).toContain('line0');
    expect(sent).toContain('line9');
    expect(sent).not.toContain('line10');
  });

  it('returns null for invalid JSON or missing fields', async () => {
    for (const reply of ['not json', '{"meaning":"only"}', '{"meaning":1,"cause":"c","fix":"f"}', null]) {
      const core = new AiCore({ provider: () => provider(reply), consent: async () => true, language: () => 'ko' });
      expect(await core.explainError(err, ''), String(reply)).toBeNull();
    }
  });

  it('accepts JSON wrapped in prose or a code fence', async () => {
    const core = new AiCore({ provider: () => provider('Sure!\n```json\n{"meaning":"m","cause":"c","fix":"f"}\n```'), consent: async () => true, language: () => 'en' });
    expect(await core.explainError(err, '')).toEqual({ meaning: 'm', cause: 'c', fix: 'f' });
  });

  it('rejects a doctor rewrite that invents numbers not in the facts', async () => {
    const facts = { what: '검증 loss가 3번째 에폭 이후 42% 올라갔어요.', why: '외우고 있어요.', try: 'Dropout을 넣어요.' };
    const bad = new AiCore({ provider: () => provider('{"what":"검증 loss가 7번째 에폭부터 99% 올랐어요.","why":"w","try":"t"}'), consent: async () => true, language: () => 'ko' });
    expect(await bad.polishDoctor('overfit', facts)).toBeNull();
    const good = new AiCore({ provider: () => provider('{"what":"3번째 에폭 뒤로 검증 loss가 42% 올랐어요.","why":"w","try":"t"}'), consent: async () => true, language: () => 'ko' });
    expect(await good.polishDoctor('overfit', facts)).toEqual({ what: '3번째 에폭 뒤로 검증 loss가 42% 올랐어요.', why: 'w', try: 't' });
  });

  it('times out slow providers', async () => {
    vi.useFakeTimers();
    const slow = provider(() => new Promise(() => {}));
    const core = new AiCore({ provider: () => slow, consent: async () => true, language: () => 'ko' }, 15_000);
    const pending = core.explainError(err, '');
    await vi.advanceTimersByTimeAsync(15_100);
    expect(await pending).toBeNull();
    vi.useRealTimers();
  });

  it('a provider that throws yields null', async () => {
    const core = new AiCore({ provider: () => provider(async () => { throw new Error('401'); }), consent: async () => true, language: () => 'ko' });
    expect(await core.explainError(err, '')).toBeNull();
  });

  it('asks for the answer in the selected language', async () => {
    const p = provider('{"meaning":"m","cause":"c","fix":"f"}');
    await new AiCore({ provider: () => p, consent: async () => true, language: () => 'en' }).explainError(err, '');
    expect(p.calls[0].system).toContain('English');
    const q = provider('{"meaning":"m","cause":"c","fix":"f"}');
    await new AiCore({ provider: () => q, consent: async () => true, language: () => 'ko' }).explainError(err, '');
    expect(q.calls[0].system).toContain('한국어');
  });
});

describe('AiCore consent modes', () => {
  it('doctor polishing never opens a consent dialog; error questions do', async () => {
    const modes: boolean[] = [];
    const core = new AiCore({ provider: () => provider('{"what":"a","why":"b","try":"c","meaning":"m","cause":"c","fix":"f"}'), consent: async (_p, interactive) => (modes.push(interactive), true), language: () => 'ko' });
    await core.polishDoctor('nan', { what: 'x', why: 'y', try: 'z' });
    await core.explainError(err, '');
    expect(modes).toEqual([false, true]);
  });
});
