export interface IssueInfo {
  title: string;
  intro: string;
  extVersion: string;
  vscodeVersion: string;
  os: string;
  python: string | null;
  torch: string | null;
  tensorflow: string | null;
  error?: string;
}

const REPO_ISSUES = 'https://github.com/mrpark10/DeepFriend/issues/new';

/** Replace home-directory paths (which contain the username) with `~`. */
export function maskPaths(text: string): string {
  return text
    .replace(/[A-Za-z]:\\Users\\[^\\\s]+/g, '~')
    .replace(/\/Users\/[^/\s]+/g, '~')
    .replace(/\/home\/[^/\s]+/g, '~');
}

export function buildIssueUrl(info: IssueInfo): string {
  const lines = [
    info.intro,
    '',
    '',
    '---',
    `- DeepFriend: ${info.extVersion}`,
    `- VS Code: ${info.vscodeVersion}`,
    `- OS: ${info.os}`,
    `- Python: ${info.python ?? '-'}`,
    `- torch ${info.torch ?? '-'} / tensorflow ${info.tensorflow ?? '-'}`,
  ];
  if (info.error) {
    lines.push('', '```', maskPaths(info.error).slice(0, 3000), '```');
  }
  const body = lines.join('\n');
  const u = new URL(REPO_ISSUES);
  u.searchParams.set('title', info.title);
  u.searchParams.set('body', body);
  return u.toString();
}
