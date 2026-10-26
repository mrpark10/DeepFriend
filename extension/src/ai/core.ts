// Optional AI wording. Decisions are always made by rules/dictionaries; the AI only
// rephrases or explains unknown errors, after the user agreed to what is sent.

export interface Provider {
  complete(system: string, user: string, signal: AbortSignal): Promise<string | null>;
}

export interface AiDeps {
  provider(): Provider | null;
  /**
   * Resolve true to allow sending `preview`. `interactive` is true only for an explicit
   * user action (a button); background polishing must not open dialogs.
   */
  consent(preview: string, interactive: boolean): Promise<boolean>;
  language(): 'ko' | 'en';
}

export interface ErrorExplanation {
  meaning: string;
  cause: string;
  fix: string;
}

export interface DoctorText {
  what: string;
  why: string;
  try: string;
}

interface ErrorLike {
  type: string;
  message: string;
}

const MAX_CODE_LINES = 10;
const MAX_FIELD = 600;

function systemPrompt(lang: 'ko' | 'en', fields: string[]): string {
  const out = `{${fields.map((f) => `"${f}": "..."`).join(', ')}}`;
  if (lang === 'ko') {
    return [
      '너는 딥러닝을 처음 배우는 고등학생·대학생에게 설명하는 친절한 선생님이야.',
      '쉬운 한국어 해요체로, 각 항목은 3문장 이내로 써.',
      '주어진 내용에 없는 숫자나 사실을 지어내지 마. 모르면 짐작이라고 말해.',
      `다른 말 없이 JSON 하나만 출력해: ${out}`,
    ].join('\n');
  }
  return [
    'You are a friendly teacher explaining deep learning to high-school and university beginners.',
    'Write plain, simple English, at most 3 sentences per field.',
    'Never invent numbers or facts that are not in the given text. If unsure, say it is a guess.',
    `Output only one JSON object and nothing else: ${out}`,
  ].join('\n');
}

function numbers(text: string): Set<string> {
  return new Set((text.match(/\d+(?:\.\d+)?/g) ?? []).map((n) => String(Number(n))));
}

function extractJson(reply: string): Record<string, unknown> | null {
  const start = reply.indexOf('{');
  const end = reply.lastIndexOf('}');
  if (start < 0 || end <= start) {
    return null;
  }
  try {
    const v = JSON.parse(reply.slice(start, end + 1)) as unknown;
    return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

function pickFields<K extends string>(obj: Record<string, unknown> | null, keys: K[]): Record<K, string> | null {
  if (!obj) {
    return null;
  }
  const out = {} as Record<K, string>;
  for (const k of keys) {
    const v = obj[k];
    if (typeof v !== 'string' || !v.trim()) {
      return null;
    }
    out[k] = v.trim().slice(0, MAX_FIELD);
  }
  return out;
}

export class AiCore {
  constructor(private deps: AiDeps, private timeoutMs = 15_000) {}

  available(): boolean {
    return this.deps.provider() !== null;
  }

  private async ask(system: string, user: string, interactive: boolean): Promise<string | null> {
    const provider = this.deps.provider();
    if (!provider) {
      return null;
    }
    if (!(await this.deps.consent(user, interactive))) {
      return null;
    }
    const ctrl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<null>((resolve) => {
      timer = setTimeout(() => {
        ctrl.abort();
        resolve(null);
      }, this.timeoutMs);
    });
    try {
      return await Promise.race([provider.complete(system, user, ctrl.signal).catch(() => null), timeout]);
    } finally {
      clearTimeout(timer);
    }
  }

  /** Explain an error that is not in the offline dictionary. */
  async explainError(err: ErrorLike, code: string): Promise<ErrorExplanation | null> {
    const lang = this.deps.language();
    const snippet = code.split('\n').slice(0, MAX_CODE_LINES).join('\n');
    const user = (lang === 'ko' ? '이 파이썬 에러를 설명해 줘.\n\n에러:\n' : 'Explain this Python error.\n\nError:\n') +
      `${err.type}: ${err.message}` + (snippet.trim() ? (lang === 'ko' ? '\n\n관련 코드:\n' : '\n\nRelated code:\n') + snippet : '');
    const reply = await this.ask(systemPrompt(lang, ['meaning', 'cause', 'fix']), user, true);
    const fields = pickFields(reply ? extractJson(reply) : null, ['meaning', 'cause', 'fix']);
    if (!fields) {
      return null;
    }
    const allowed = numbers(user);
    for (const v of Object.values(fields)) {
      for (const n of numbers(v)) {
        if (!allowed.has(n)) {
          return null;
        }
      }
    }
    return fields;
  }

  /** Rephrase a rule-based doctor card in easier words. Numbers must stay exactly the same. */
  async polishDoctor(id: string, text: DoctorText): Promise<DoctorText | null> {
    const lang = this.deps.language();
    const user = (lang === 'ko' ? `학습 진단(${id}) 문장을 더 쉽게 다시 써 줘. 숫자는 그대로 둬.\n\n` : `Rewrite this training diagnosis (${id}) more simply. Keep every number.\n\n`) +
      JSON.stringify(text, null, 2);
    const reply = await this.ask(systemPrompt(lang, ['what', 'why', 'try']), user, false);
    const fields = pickFields(reply ? extractJson(reply) : null, ['what', 'why', 'try']);
    if (!fields) {
      return null;
    }
    const allowed = numbers(text.what + ' ' + text.why + ' ' + text.try);
    for (const v of Object.values(fields)) {
      for (const n of numbers(v)) {
        if (!allowed.has(n)) {
          return null;
        }
      }
    }
    return fields;
  }
}
