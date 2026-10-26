import { en } from './en';
import { ko, StringKey } from './ko';

export type Language = 'ko' | 'en';
export type { StringKey };

let current: Language = 'ko';

export function setLanguage(lang: Language): void {
  current = lang === 'en' ? 'en' : 'ko';
}

export function getLanguage(): Language {
  return current;
}

/** Look up a message in the current language and fill `{name}` placeholders. */
export function t(key: StringKey, args?: Record<string, string | number>): string {
  const table: Record<StringKey, string> = current === 'en' ? en : ko;
  const template = table[key] ?? ko[key] ?? key;
  if (!args) {
    return template;
  }
  return template.replace(/\{([a-zA-Z0-9_]+)\}/g, (m, name: string) => (name in args ? String(args[name]) : m));
}

/** Pick the right side of a bilingual pair. */
export function pick(pair: { ko: string; en: string }): string {
  return current === 'en' ? pair.en : pair.ko;
}
