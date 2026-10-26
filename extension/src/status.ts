import * as vscode from 'vscode';
import { t } from './i18n/i18n';
import type { Status } from './server/client';

const ICON: Record<Status, string> = {
  starting: '$(sync~spin)',
  ready: '$(check)',
  busy: '$(pulse)',
  error: '$(error)',
  noPython: '$(warning)',
};

const KEY = {
  starting: 'status.starting',
  ready: 'status.ready',
  busy: 'status.busy',
  error: 'status.error',
  noPython: 'status.noPython',
} as const;

export class StatusBar implements vscode.Disposable {
  private item = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  private state: Status = 'starting';
  private framework: 'pytorch' | 'tensorflow' | null = null;

  constructor() {
    this.item.command = 'deepfriend.showStatus';
    this.item.name = 'DeepFriend';
    this.render();
  }

  set(state: Status): void {
    this.state = state;
    this.render();
  }

  get(): Status {
    return this.state;
  }

  setFramework(fw: 'pytorch' | 'tensorflow' | null): void {
    this.framework = fw;
    this.render();
  }

  render(): void {
    const fw = this.framework === 'pytorch' ? ' · PyTorch' : this.framework === 'tensorflow' ? ' · TensorFlow' : '';
    this.item.text = `${ICON[this.state]} ${t(KEY[this.state])}${fw}`;
    this.item.tooltip = t('status.tooltip');
    this.item.backgroundColor = this.state === 'error' ? new vscode.ThemeColor('statusBarItem.errorBackground') : undefined;
    if (this.framework || this.state === 'error' || this.state === 'noPython') {
      this.item.show();
    } else {
      this.item.hide();
    }
  }

  dispose(): void {
    this.item.dispose();
  }
}
