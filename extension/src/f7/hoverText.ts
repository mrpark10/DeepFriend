import type { Level } from '../config';
import { pick, t } from '../i18n/i18n';
import type { Concept } from './concepts';

/** Markdown for a concept hover. The hovered parameter (if any) comes first. */
export function hoverMarkdown(c: Concept, level: Level, param?: string, shape?: string): string {
  const parts: string[] = [];
  if (param && c.params[param]) {
    parts.push(`**\`${param}\`** — ${pick(c.params[param])}`, '---');
  }
  parts.push(`**${pick(c.title)}**`, pick(level === 'normal' ? c.what.normal : c.what.beginner));
  if (shape) {
    parts.push(t('hover.shape', { shape }));
  }
  if (c.diagram) {
    parts.push('```text\n' + pick(c.diagram) + '\n```');
  }
  const names = Object.keys(c.params);
  if (names.length) {
    parts.push(`**${t('hover.params')}**\n\n` + names.map((n) => `- \`${n}\`: ${pick(c.params[n])}`).join('\n'));
  }
  parts.push(`[${t('hover.doc')}](${c.doc})`);
  return parts.join('\n\n');
}
