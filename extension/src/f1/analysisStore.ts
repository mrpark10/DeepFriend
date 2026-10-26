import { FRAMEWORK_IMPORT } from '../frameworkImport';
import { Emitter } from '../server/client';
import type { AnalysisResult } from '../server/types';

export interface DocLike {
  uri: { toString(): string };
  version: number;
  languageId: string;
  getText(): string;
}

export type AnalyzeFn = (uri: string, text: string) => Promise<AnalysisResult | null>;

/**
 * Decides when to analyze which document and keeps the latest result per uri.
 * Typing is debounced; saving analyzes immediately (spec: "저장 후 1초 안").
 * A response is applied only if the document has not changed since the request.
 */
export class AnalysisStore {
  private results = new Map<string, AnalysisResult>();
  private timers = new Map<string, ReturnType<typeof setTimeout>>();
  private versions = new Map<string, number>();
  private emitter = new Emitter<string>();
  readonly onDidChange = this.emitter.event;

  constructor(private analyze: AnalyzeFn, private debounceMs = 400) {}

  get(uri: string): AnalysisResult | undefined {
    return this.results.get(uri);
  }

  static relevant(doc: DocLike): boolean {
    return doc.languageId === 'python' && FRAMEWORK_IMPORT.test(doc.getText());
  }

  trackVersion(doc: DocLike): void {
    this.versions.set(doc.uri.toString(), doc.version);
  }

  onOpen(doc: DocLike): void {
    this.schedule(doc, 0);
  }

  onChange(doc: DocLike): void {
    this.schedule(doc, this.debounceMs);
  }

  onSave(doc: DocLike): void {
    this.schedule(doc, 0);
  }

  onClose(doc: DocLike): void {
    const uri = doc.uri.toString();
    this.cancel(uri);
    this.versions.delete(uri);
    if (this.results.delete(uri)) {
      this.emitter.fire(uri);
    }
  }

  private cancel(uri: string): void {
    const t = this.timers.get(uri);
    if (t) {
      clearTimeout(t);
      this.timers.delete(uri);
    }
  }

  private schedule(doc: DocLike, delay: number): void {
    const uri = doc.uri.toString();
    this.trackVersion(doc);
    this.cancel(uri);
    if (doc.languageId !== 'python') {
      return;
    }
    if (!AnalysisStore.relevant(doc)) {
      if (this.results.delete(uri)) {
        this.emitter.fire(uri);
      }
      return;
    }
    const text = doc.getText();
    const version = doc.version;
    this.timers.set(
      uri,
      setTimeout(() => {
        this.timers.delete(uri);
        void this.run(uri, text, version);
      }, delay),
    );
  }

  private async run(uri: string, text: string, version: number): Promise<void> {
    let res: AnalysisResult | null;
    try {
      res = await this.analyze(uri, text);
    } catch {
      return;
    }
    if (!res || this.versions.get(uri) !== version) {
      return;
    }
    this.results.set(uri, res);
    this.emitter.fire(uri);
  }
}
