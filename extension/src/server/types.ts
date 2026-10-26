// Mirrors the dicts produced by python/deepfriend/shapes/analyze.py.
// Lines and columns are 0-based; columns are UTF-16 code units (VS Code positions).

export interface InitializeResult {
  version: string;
  python: string;
  pythonVersion: string;
  torch: string | null;
  tensorflow: string | null;
}

export type DiagCode =
  | 'linear_in'
  | 'conv_in'
  | 'bn_features'
  | 'reshape_numel'
  | 'cat_mismatch'
  | 'nonpositive'
  | 'keras_rank'
  | 'rank'
  | 'invalid_arg'
  | 'broadcast'
  | 'runtime_error';

export interface Fix {
  line: number;
  col: number;
  endLine: number;
  endCol: number;
  newText: string;
}

export interface Diag {
  line: number;
  col: number;
  endLine: number;
  endCol: number;
  code: DiagCode;
  args: Record<string, string | number>;
  fix?: Fix;
}

export interface Hint {
  line: number;
  shape: string;
  all?: string[];
  unknownReason?: string;
}

export interface Block {
  name: string;
  kind: string;
  line: number;
  outShape: string;
  params: number | null;
  error: boolean;
}

export interface Memory {
  params: number;
  grads: number;
  optimizer: number;
  activations: number;
  total: number;
}

export interface ModelInfo {
  name: string;
  kind: 'torch' | 'keras';
  line: number;
  endLine: number;
  input: { line: number; shapes: string[] } | null;
  needsInput: boolean;
  suggestedInput: string | null;
  hints: Hint[];
  diagnostics: Diag[];
  blocks: Block[];
  totalParams: number | null;
  memory: Memory | null;
  verified: boolean;
}

export interface AnalysisResult {
  framework: 'pytorch' | 'tensorflow' | null;
  models: ModelInfo[];
  syntaxError: { line: number; col: number; msg: string } | null;
}

export interface ArgLocation {
  line: number;
  col: number;
  endCol: number;
  value: number | null;
  insert: boolean;
  defaultValue?: number;
}
