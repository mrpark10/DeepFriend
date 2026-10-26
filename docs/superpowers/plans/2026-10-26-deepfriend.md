# DeepFriend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **The user chose executing-plans, inline, no subagents.**

**Goal:** Build the DeepFriend VS Code extension (F1–F7) plus its bundled stdlib-only Python package, QA it thoroughly, and ship to VS Code Marketplace, Open VSX and PyPI.

**Architecture:** TypeScript extension (display only) spawns a bundled Python analysis server over stdio JSON lines. Training-side `deepfriend` package writes runs to `.deepfriend/runs/<id>/` which the extension watches. All user-facing copy lives in the TypeScript i18n tables; Python returns codes + args.

**Tech Stack:** TypeScript 5, esbuild, VS Code API ^1.93, vitest (pure TS), @vscode/test-cli + mocha (integration), uPlot (charts, bundled), Python ≥3.9 stdlib (`ast`, `json`, `subprocess`), pytest, torch 2.x / tensorflow 2.x (optional at runtime, required in dev venv for parity tests), litert-torch (F6).

**Spec:** `docs/superpowers/specs/2026-10-26-deepfriend-design.md`

## Global Constraints

- Names: extension `mrpark10.deepfriend`, display name `DeepFriend`, Python package `deepfriend`, run folder `.deepfriend/`, command prefix `DeepFriend: …`, input comment `# df: input (…)` (also accept `# dm: input`).
- `engines.vscode`: `^1.93.0`. Python support `>=3.9`. torch 2.x, tensorflow 2.x. Windows/macOS/Linux.
- `python/deepfriend` imports only the stdlib at module level; torch/tensorflow/keras imported lazily inside functions and every such import is guarded.
- Never execute the user's whole file. Runtime check executes only: imports of `torch, torchvision, math, numpy, typing, collections, functools, dataclasses` (and their submodules), module-level assignments whose value is a literal, and class definitions.
- Shape display format: `→ (32, 64, 112, 112)`; unknown is `→ ?`; Keras batch shows `None`.
- Every user-visible string exists in both `ko` and `en`; default language `ko`.
- No code edit without a preview + user confirmation. No network call without consent (`deepfriend.ai.consent`).
- Speed: static analysis ≤100 ms for a 1000-line file; hints refreshed ≤1 s after save; doctor graph refresh every 1 s.
- `watch()` must never raise into user code; all collector code is wrapped in try/except.
- Contact: sk92977852@gmail.com. Repo: `mrpark10/DeepFriend` (private until launch). License MIT, © 2026 박준호 (Junho Park).
- Commit after every task (`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).
- Deviation from spec doc recorded here: doctor rules run in TypeScript (`extension/src/f2/doctorRules.ts`), not `doctor.py` — the extension already reads the metrics files, so no IPC round-trip is needed. Rules are still deterministic code, AI only rewrites wording.

## Review Focus

1. **Files with syntax errors mid-typing** — analysis must return the last good result (hints stay, no crash, no flood of errors). Test in Task 7 (`test_syntax_error_keeps_previous`).
2. **Constructor args without defaults / layers built in loops / `if` branches** — must show `?`, never a wrong number. Tests in Task 4 (`test_unknown_ctor_arg_is_question`, `test_if_branches_disagree_unknown`).
3. **Windows paths & `py` launcher, paths with spaces/Korean characters** (e.g. `C:\Users\박준호\딥러닝 과제\train.py`) — finder, server spawn, run folder, traceback parsing. Tests in Task 8 (`findPython` on win32 candidates) and Task 11 (`parses windows traceback with korean path`).
4. **User code that is slow/infinite/huge in `__init__`** (e.g. `torch.randn(10**10)`, `while True`) during runtime check — must time out at 3 s and keep static result. Test in Task 6 (`test_runtime_timeout_falls_back`).
5. **Training run killed (Ctrl+C / crash / OOM) mid-write** — run shows `stopped`/`failed`, partial last JSONL line is ignored. Tests in Task 13 (`test_keyboard_interrupt_marks_stopped`) and Task 15 (`ignores truncated last line`).

---

## File Structure

```
package.json (root: workspaces scripts only)
extension/
  package.json  tsconfig.json  esbuild.mjs  vitest.config.ts  .vscode-test.mjs  .vscodeignore
  package.nls.json (ko)  package.nls.en.json
  media/ icon.png  deepfriend.svg  webview/{common.css, doctor.js, model.js, compare.js, card.js, samples.js}  vendor/uPlot.*
  walkthrough/*.md
  examples/ (copied from /examples at build)
  python/   (copied from /python/deepfriend at build; gitignored)
  src/
    extension.ts
    i18n/{i18n.ts, ko.ts, en.ts}
    python/findPython.ts
    server/{client.ts, protocol.ts, types.ts}
    status.ts  config.ts  report.ts  terminalEnv.ts
    f1/{analysisStore.ts, inlayHints.ts, diagnostics.ts, codeActions.ts, codeLens.ts, setInput.ts}
    f2/{doctorRules.ts, runReader.ts, runWatcher.ts, doctorView.ts, actions.ts}
    f3/{traceback.ts, errorDictionary.ts, terminalWatcher.ts, notebookWatcher.ts, errorCard.ts}
    f4/modelView.ts
    f5/{runsTree.ts, compare.ts, restore.ts}
    f6/exportCommand.ts
    f7/{concepts.ts, hover.ts}
    ai/{ai.ts, consent.ts, providers.ts}
    webview/html.ts
    test/unit/*.test.ts   test/suite/*.test.ts   test/fixtures/*
python/
  pyproject.toml
  deepfriend/
    __init__.py  protocol.py  server.py  findargs.py
    shapes/{__init__.py, dims.py, torch_formulas.py, torch_layers.py, static_torch.py,
            keras_layers.py, static_keras.py, annotations.py, extract.py, runtime_torch.py, analyze.py}
    watch/{__init__.py, run.py, torch_hooks.py, keras_cb.py, samples.py, snapshot.py, excepthook.py}
    export/{__init__.py, cli.py, convert.py, verify.py, flutter.py}
  tests/...
examples/ 01_mnist_mlp.py 02_cifar_cnn.py 03_keras_cnn.py 04_shape_error.py 05_doctor/{nan,overfit,plateau,oscillation,chance,healthy}.py
docs/ guide.md faq.md privacy.md privacy.en.md terms.md terms.en.md publishing.md
.github/workflows/{ci.yml, release.yml}
README.md README.en.md CHANGELOG.md LICENSE
```

## Shared data contracts (used by many tasks)

**Server protocol** (`python/deepfriend/protocol.py`, `extension/src/server/protocol.ts`): one JSON object per line on stdout/stdin, UTF-8. Request `{"id": int, "method": str, "params": {}}`; response `{"id": int, "result": any}` or `{"id": int, "error": {"message": str, "trace": str}}`. Methods: `initialize` → `{version, python, pythonVersion, torch: str|null, tensorflow: str|null}`; `analyze {uri, text, runtime: bool}` → `AnalysisResult`; `findArgument {text, kind: "lr"|"batch_size"}` → `ArgLocation[]`; `shutdown` → `null`. stderr is log only.

**AnalysisResult** (`extension/src/server/types.ts` mirrors Python dicts; lines/cols 0-based, cols in UTF-16 code units — Python converts from its str indices):
```ts
interface AnalysisResult { framework: "pytorch"|"tensorflow"|null; models: ModelInfo[]; syntaxError: {line:number; col:number; msg:string}|null }
interface ModelInfo { name: string; kind: "torch"|"keras"; line: number; endLine: number;
  input: {line:number; shapes:string[]}|null; needsInput: boolean; suggestedInput: string|null;
  hints: Hint[]; diagnostics: Diag[]; blocks: Block[]; totalParams: number|null;
  memory: {params:number; grads:number; optimizer:number; activations:number; total:number}|null;
  verified: boolean }
interface Hint { line: number; shape: string; all?: string[]; unknownReason?: string }
interface Diag { line:number; col:number; endLine:number; endCol:number; code: DiagCode; args: Record<string, string|number>;
  fix?: {line:number; col:number; endLine:number; endCol:number; newText:string} }
type DiagCode = "linear_in"|"conv_in"|"bn_features"|"reshape_numel"|"cat_mismatch"|"nonpositive"|"keras_rank"|"runtime_error"
interface Block { name:string; kind:string; line:number; outShape:string; params:number|null; error:boolean }
interface ArgLocation { line:number; col:number; endCol:number; value:number|null; insert:boolean; defaultValue?: number }
```
`shape` strings: `"(32, 64, 112, 112)"`, `"(None, 10)"`, `"?"`.

**Run folder** `.deepfriend/runs/<YYYYMMDD-HHMMSS-xxxx>/`:
- `meta.json` `{id, script, cwd, startedAt, endedAt|null, status: "running"|"finished"|"failed"|"stopped", pid, framework, config:{lr?, batch_size?, optimizer?, epochs?, ...user}, model:{name?, params?}, numClasses|null, final:{loss?,val_loss?,acc?,val_acc?}, best:{...same, epoch?}, starred:false, deepfriendVersion, git:str|null, record: bool}`
- `metrics.jsonl` lines `{"t":float,"step":int,"epoch":int,"split":"train"|"val","loss"?:float|null,"acc"?:float,"lr"?:float}` (NaN written as `null` with `"nan":true`)
- `samples.json` `{items:[{shape:[h,w,c]|[n], pixels:number[] (0-255, ≤32x32), label:str|int}]}`
- `code/<relative path>` snapshots; `error.json` `{type, message, traceback, frames:[{file,line,name}]}`
- `.deepfriend/.gitignore` containing `*`.

---

### Task 0: Dev environment and skeleton

**Files:** Create: `package.json`, `.gitignore`, `extension/{package.json,tsconfig.json,esbuild.mjs,vitest.config.ts,.vscode-test.mjs,.vscodeignore}`, `extension/src/extension.ts`, `python/pyproject.toml`, `python/deepfriend/__init__.py`, `python/tests/test_smoke.py`, `extension/src/test/unit/smoke.test.ts`, `Makefile`.

- [ ] **Step 1:** `brew install uv`; `uv venv .venv --python 3.12`; `uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu`; `uv pip install --python .venv tensorflow pytest`; `uv venv .venv39 --python 3.9 && uv pip install --python .venv39 pytest`. Verify: `.venv/bin/python -c "import torch,tensorflow;print(torch.__version__,tensorflow.__version__)"` prints two 2.x versions.
- [ ] **Step 2:** Root `.gitignore`: `.venv*/ node_modules/ extension/out/ extension/dist/ extension/python/ extension/examples/ *.vsix .deepfriend/ __pycache__/ .vscode-test/`.
- [ ] **Step 3:** `extension/package.json`: name `deepfriend`, publisher `mrpark10`, version `0.1.0`, `engines.vscode ^1.93.0`, `main ./dist/extension.js`, `activationEvents: ["onLanguage:python","onNotebook:jupyter-notebook","workspaceContains:.deepfriend"]`, `l10n` not used; scripts `build` (copies `../python/deepfriend` → `python/deepfriend` excluding `tests`/`__pycache__`, copies `../examples` → `examples`, then esbuild), `watch`, `test:unit` (vitest run), `test:int` (vscode-test), `package` (vsce package). devDeps: typescript, esbuild, vitest, @types/vscode@1.93.0, @types/node, @vscode/test-cli, @vscode/test-electron, mocha, @types/mocha, @vscode/vsce, ovsx. deps: `@vscode/python-extension`, `uplot`.
- [ ] **Step 4:** `python/pyproject.toml`: name `deepfriend`, version `0.1.0`, `requires-python >=3.9`, no dependencies, MIT, readme `../README.md` is not allowed outside dir → use `python/README.md` (short, links to repo). `deepfriend/__init__.py` has `__version__ = "0.1.0"`.
- [ ] **Step 5:** Smoke tests: `python/tests/test_smoke.py::test_version` asserts `deepfriend.__version__ == "0.1.0"`; `smoke.test.ts` asserts `1+1===2`. Integration smoke `src/test/suite/activate.test.ts`: extension `mrpark10.deepfriend` activates after opening a `.py` doc.
- [ ] **Step 6:** Run `cd extension && npm i && npm run build && npm run test:unit && npm run test:int`; `.venv/bin/pytest python -q`; `.venv39/bin/pytest python -q`. All PASS.
- [ ] **Step 7:** Commit `chore: project skeleton`.

### Task 1: Python protocol and server skeleton

**Files:** Create `python/deepfriend/protocol.py`, `python/deepfriend/server.py`; Test `python/tests/test_server.py`.

**Interfaces:** Produces `protocol.read_messages(stream) -> Iterator[dict]`, `protocol.write_message(stream, obj) -> None` (thread-safe lock, `ensure_ascii=False`, flush); `server.main() -> None`; `server.Server.register(method: str, fn: Callable[[dict], Any])`. `python -m deepfriend.server` runs `main()`.

- [ ] **Step 1: Failing tests** — spawn `sys.executable -m deepfriend.server` with `PYTHONPATH=python`:
  - `test_initialize_reports_versions`: result has `version=="0.1.0"`, `pythonVersion` startswith `"3."`, keys `torch` and `tensorflow` present (values str or None — detect with `importlib.util.find_spec` + `importlib.metadata.version`, never import torch here).
  - `test_unknown_method_returns_error`: `{"id":2,"method":"nope"}` → response has `error.message` containing `"nope"`, server keeps running (next `initialize` works).
  - `test_bad_json_line_ignored`: write `"{not json\n"` then a valid request → valid response arrives.
  - `test_shutdown_exits`: process exits code 0 within 2 s.
  - `test_stdin_eof_exits`: closing stdin exits within 2 s.
- [ ] **Step 2:** Run `.venv/bin/pytest python/tests/test_server.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement. Requests handled on the main thread in order, except `analyze` which later tasks run with "latest wins" (Task 7). Exceptions → error response with `traceback.format_exc()`.
- [ ] **Step 4:** Tests PASS on `.venv` and `.venv39`.
- [ ] **Step 5:** Commit `feat(py): stdio JSON server`.

### Task 2: PyTorch shape formulas

**Files:** Create `python/deepfriend/shapes/dims.py`, `python/deepfriend/shapes/torch_formulas.py`; Test `python/tests/shapes/test_torch_formulas.py`, `python/tests/shapes/test_torch_parity.py`.

**Interfaces:** Produces in `dims.py`: `Dim = Union[int, str]` (str = symbolic like `"B"`), `Shape = Optional[Tuple[Dim, ...]]` (None = unknown), `fmt_shape(shape) -> str`, `numel(shape) -> Optional[int]`, `broadcast(a, b) -> Shape | ShapeError`. `class ShapeError(Exception): code: str; args: dict`.
In `torch_formulas.py`: one function per op, signature `fn(in_shape: Shape, **params) -> Shape` raising `ShapeError`: `linear(in_features, out_features)`, `conv(nd, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1)`, `pool(nd, kind:"max"|"avg", kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False)`, `adaptive_pool(nd, output_size)`, `flatten(start_dim=1, end_dim=-1)`, `batchnorm(nd, num_features)`, `layernorm(normalized_shape)`, `embedding(num_embeddings, embedding_dim)`, `identity()`, `reshape(target: tuple)`, `cat(shapes: list, dim)`, `squeeze(dim=None)`, `unsqueeze(dim)`, `permute(dims)`, `transpose(d0, d1)`, `reduce(dim, keepdim=False)`.

Pinned rules (from PyTorch docs):
- conv out = `floor((L + 2p − d·(k−1) − 1)/s + 1)`; `padding="same"` keeps L (stride must be 1), `"valid"` = 0; int-or-tuple params; input rank nd+2 or nd+1 (unbatched).
- pool: `stride=None` → kernel_size; ceil_mode uses ceil, then if `(out−1)·s ≥ L + p` then `out −= 1`; avg pool has no dilation.
- Errors: `linear_in {expected, actual}` when last dim ≠ in_features; `conv_in {expected, actual}`; `bn_features {expected, actual}`; `reshape_numel {from, to}`; `cat_mismatch {dim, a, b}`; `nonpositive {layer}` when any output dim < 1.
- Symbolic dims propagate untouched through batch axes; a formula needing a symbolic spatial dim returns `?` for that dim (keep the string `"?"` inside the tuple is NOT allowed — return None for whole shape).

- [ ] **Step 1: Failing unit tests** with exact values, including: `conv(2,3,64,7,stride=2,padding=3)` on `(32,3,224,224)` → `(32,64,112,112)`; `pool(2,"max",2)` on `(32,64,112,112)` → `(32,64,56,56)`; `pool(2,"max",3,stride=2,ceil_mode=True)` on `(1,1,6,6)` → `(1,1,3,3)` and on `(1,1,5,5)` padding=1 → `(1,1,3,3)`; `linear(10000,10)` on `(32,12544)` raises `ShapeError("linear_in", expected=10000, actual=12544)`; `reshape((-1,))` on `(2,3,4)` → `(24,)`; `reshape((5,-1))` on `(2,3,4)` raises `reshape_numel`; `cat([(2,3),(2,5)],1)` → `(2,8)`; `cat([(2,3),(4,5)],1)` raises `cat_mismatch`; `conv(2,1,1,5)` on `(1,1,3,3)` raises `nonpositive`; symbolic `("B",3,32,32)` through conv k=3 → `("B",16,30,30)`.
- [ ] **Step 2: Parity test** `test_torch_parity.py` (skip if no torch): seeded `random.Random(0)`, 400 random configs per op family (conv1d/2d incl. groups/dilation/padding str, max/avg pool incl. ceil_mode/padding, adaptive, flatten ranges, reshape with -1, cat), compute with formula and with real `torch.nn` on CPU zero tensors (small sizes ≤40); assert identical shape, and assert formula raises iff torch raises.
- [ ] **Step 3:** Run → FAIL. **Step 4:** Implement. **Step 5:** Run both files → PASS (0 mismatches). **Step 6:** Commit `feat(py): torch shape formulas with parity tests`.

### Task 3: PyTorch layer parsing and parameter counts

**Files:** Create `python/deepfriend/shapes/torch_layers.py`; Test `python/tests/shapes/test_torch_layers.py`.

**Interfaces:**
- Consumes: Task 2 formulas.
- Produces: `@dataclass Layer(kind: str, params: dict, line: int, col: int, end_line: int, end_col: int, arg_ranges: dict[str, tuple[int,int,int,int]], children: list[Layer] | None)`; `parse_layer(call: ast.Call, consts: dict) -> Optional[Layer]` (recognises `nn.X`, `torch.nn.X`, bare `X` when imported from torch.nn; positional args mapped to the official parameter order; `nn.Sequential(...)` → children, `nn.ModuleList([...])` → children; returns `Layer(kind="unknown")` for other `nn.*` and None for non-nn calls); `infer(layer, in_shape) -> Shape` (raises ShapeError); `param_count(layer) -> Optional[int]`; `ACTIVATIONS: frozenset` (ReLU, ReLU6, LeakyReLU, PReLU, ELU, SELU, CELU, GELU, SiLU, Mish, Sigmoid, Tanh, Hardtanh, Hardswish, Hardsigmoid, Softplus, Softsign, Softmax, LogSoftmax, Softmin, Tanhshrink, Threshold, Dropout, Dropout1d, Dropout2d, Dropout3d, AlphaDropout, Identity).
- Param counts (match `sum(p.numel() for p in m.parameters())`): Linear `in·out + out·bias`; Conv `out·(in/groups)·∏k + out·bias`; BatchNorm `2C` if affine else 0; LayerNorm `2·∏shape` if elementwise_affine (bias param respected); Embedding `n·d`; PReLU `num_parameters` (default 1); activations/pool/flatten/dropout 0; unknown → None.
- `consts` maps module-level and `__init__`-param names to literal values (ints, tuples) so `nn.Linear(hidden, 10)` resolves when `hidden` is known; unresolved param → stored as `None` and inference returns unknown.

- [ ] **Step 1: Failing tests**: positional and keyword parsing (`nn.Conv2d(3, 64, 7, 2, 3)` == keyword form), `arg_ranges["in_features"]` points at the exact source span of `10000` in `nn.Linear(10000, 10)`, Sequential children lines are each child's line, `param_count` exact for 12 layer types; parity test (skip w/o torch) comparing `param_count` to real module for 200 random configs.
- [ ] **Step 2:** FAIL → **Step 3:** implement → **Step 4:** PASS → **Step 5:** commit `feat(py): torch layer parsing`.

### Task 4: Static PyTorch model analyzer

**Files:** Create `python/deepfriend/shapes/annotations.py`, `python/deepfriend/shapes/static_torch.py`; Test `python/tests/shapes/test_static_torch.py`, fixtures `python/tests/fixtures/torch/*.py`.

**Interfaces:**
- Consumes: Tasks 2–3.
- Produces: `annotations.find_input(lines: list[str], def_line: int) -> Optional[InputSpec]` where `InputSpec(line:int, shapes: list[Shape], raw: str)`; regex `^\s*#\s*(?:df|dm)\s*:\s*input\s+(.+?)\s*$` searched in the contiguous comment/blank/decorator block directly above `def_line`; value is one or more tuples `(…), (…)` of ints or identifiers. `annotations.suggest_input(text: str) -> Optional[str]` returns `"(B, 1, 28, 28)"` if `MNIST`/`FashionMNIST`/`KMNIST`/`EMNIST` appears, `"(B, 3, 32, 32)"` for `CIFAR10`/`CIFAR100`/`SVHN`, `"(B, 3, 224, 224)"` for `Resize(224`/`RandomResizedCrop(224`/`ImageNet`, else None.
  `static_torch.analyze_torch(tree: ast.Module, text: str) -> list[ModelInfo dict]` (dict shape = contract above; `verified=False`).
- Algorithm (pinned decisions):
  - Model = class whose bases include `nn.Module`/`torch.nn.Module`/`Module`, or another model class in the same file.
  - `__init__`: `self.<name> = <layer call | Sequential | ModuleList | user model class call>`; bind `__init__` params to their defaults for consts; non-default params → unknown.
  - `forward`: walk statements; env maps names → `Shape | int | tuple | Unknown`. Support: `self.layer(x)`, nested calls `self.pool(F.relu(self.conv(x)))`, `F.<activation>`, `F.max_pool2d/avg_pool2d/adaptive_avg_pool2d/dropout/softmax/log_softmax/linear? (no)`, `torch.relu/sigmoid/tanh/flatten/cat/stack/squeeze/unsqueeze`, tensor methods `view reshape flatten size squeeze unsqueeze permute transpose mean sum contiguous float double to cuda cpu detach relu sigmoid softmax`, `x.shape[i]`, `x.size(i)`, `x.size()`, binary ops with broadcasting, `for layer in self.<ModuleList|Sequential>:` iterated child by child, `for _ in range(<int literal>)` unrolled up to 64, `if/else` → analyse both, keep per-variable result only if equal else Unknown, `return`.
  - Hints: each assignment line in `forward` gets the shape of the assigned name (tuple unpacking → each tensor; non-tensor → no hint); each layer definition line in `__init__` (and each Sequential child line) gets the output shape(s) observed when called; multiple different shapes → `shape` = first, `all` = list.
  - Errors: ShapeError → Diag at the forward call line (range of the call) AND, when the error is `linear_in`/`conv_in`/`bn_features`, a Diag at the layer definition's argument span with `fix.newText = str(actual)`. After an error the variable becomes Unknown (analysis continues, no cascade errors).
  - Unknown reasons (for hover): `"ctor_arg"`, `"unsupported_op:<name>"`, `"loop"`, `"branch"`, `"no_input"`.
  - `needsInput = input is None`; then no hints except `?` is NOT emitted (empty hints) and CodeLens handles it.
  - Blocks: one per `self.<name>` layer in call order (Sequential expanded, name `features.0`), with `outShape`, `params`, `error`.
  - Memory: `params·4 + params·4 + params·8 + activations·4`, activations = sum of numel of every block output (requires int batch; symbolic → memory None).
- [ ] **Step 1: Failing tests** (each fixture = small realistic file, assert exact hints by line):
  - `test_spec_example_resnet_stem`: `# df: input (32, 3, 224, 224)`, conv7x7 s2 p3 → hint `(32, 64, 112, 112)` on conv line.
  - `test_linear_mismatch_diag_and_fix`: CNN flattening to 12544 into `nn.Linear(10000, 10)` → Diag code `linear_in` args `{expected:10000, actual:12544}`, fix newText `"12544"` at the `10000` span.
  - `test_sequential_children_hints`, `test_functional_and_view`, `test_cat_two_branches`, `test_modulelist_loop`, `test_nested_user_module`, `test_dm_alias_comment`, `test_multiple_inputs`, `test_symbolic_batch`.
  - `test_unknown_ctor_arg_is_question` (Review Focus 2): `def __init__(self, n): self.fc = nn.Linear(n, 10)` → fc hint `?` with `unknownReason=="ctor_arg"`, no Diag.
  - `test_if_branches_disagree_unknown` (Review Focus 2).
  - `test_no_cascade_after_error`: only one error reported for one bug.
  - `test_needs_input_and_suggestion`: no comment + `datasets.MNIST` in text → `needsInput True`, `suggestedInput "(B, 1, 28, 28)"`.
  - `test_params_and_memory` exact numbers for a LeNet.
- [ ] **Step 2:** FAIL → **Step 3:** implement → **Step 4:** PASS (`.venv` and `.venv39`) → **Step 5:** commit `feat(py): static torch analyzer`.

### Task 5: Keras formulas and static analyzer

**Files:** Create `python/deepfriend/shapes/keras_layers.py`, `python/deepfriend/shapes/static_keras.py`; Test `python/tests/shapes/test_static_keras.py`, `python/tests/shapes/test_keras_parity.py`.

**Interfaces:** Produces `static_keras.analyze_keras(tree, text) -> list[ModelInfo dict]` (`kind="keras"`). Recognises `keras.Sequential([...])`, `tf.keras.Sequential`, `Sequential(`, followed by `<name>.add(<layer>)` statements in the same scope, and functional chains starting at `keras.Input(shape=…)`/`layers.Input`. Layers: Dense (applies to last axis; params `in·units + units·use_bias`), Conv2D (`filters, kernel_size, strides=1, padding="valid"|"same", dilation_rate=1, use_bias`; requires rank 4 → else Diag `keras_rank`; same → `ceil(L/s)`, valid → `floor((L − d(k−1) − 1)/s) + 1`), MaxPooling2D (`pool_size=2, strides=None→pool_size, padding`), Flatten, Dropout, Activation, BatchNormalization (params `4C`), InputLayer/Input. Input shape from first layer `input_shape=`, `Input(shape=)`, or `# df: input` above the model assignment (Keras form: without batch → batch `None`). Model name = assigned variable.
- [ ] **Step 1: Failing tests**: classic MNIST Keras CNN exact hints `(None, 26, 26, 32)` etc.; Dense on 4D input keeps rank (no error); Conv2D after Flatten → `keras_rank`; `.add()` style; functional style; params total equals hand value.
- [ ] **Step 2: Parity** (skip w/o tf): 150 random Sequential models (Conv2D/MaxPool/Dense/Flatten combos) — compare every layer's `output.shape` and `count_params()` from real `keras` with ours; random configs that raise in Keras must produce a Diag.
- [ ] **Step 3–5:** FAIL → implement → PASS → commit `feat(py): keras static analyzer`.

### Task 6: Runtime PyTorch verification worker

**Files:** Create `python/deepfriend/shapes/extract.py`, `python/deepfriend/shapes/runtime_torch.py`; Test `python/tests/shapes/test_runtime_torch.py` (skip w/o torch).

**Interfaces:**
- Produces: `extract.safe_source(tree: ast.Module) -> ast.Module` (keeps allowed imports per Global Constraints, literal module assigns, all ClassDefs; keeps original `lineno`). `runtime_torch.Worker` — a long-lived subprocess (`python -m deepfriend.shapes.runtime_torch`) that imports torch once; `Worker.check(text: str, model: str, inputs: list[tuple[int,...]], timeout: float = 3.0) -> Optional[RuntimeResult]` where `RuntimeResult = {"lines": {line: [shape strings]}, "modules": {qualname: {"out": [shape], "params": int}}, "totalParams": int, "error": None | {"line": int, "message": str, "type": str}}`. Timeout → kill worker, restart lazily, return None. Any non-shape failure (NameError, unsupported meta op, ctor needs args) → None.
- Mechanics: compile `safe_source` with filename `"<deepfriend>"`; instantiate under `with torch.device("meta")`; symbolic dims replaced by 2 (and those lines' results are then not used to override static values for symbolic dims — only to detect errors); register forward hooks on all submodules; `sys.settrace` limited to code objects whose `co_filename == "<deepfriend>"` recording tensor-valued locals at each line event, attributed to the previously executed line in that frame, final values on `return`. Shape errors detected by message patterns (`mat1 and mat2 shapes`, `size mismatch`, `Expected .* channels`, `shape '\[.*\]' is invalid`, `Sizes of tensors must match`, `must match the size of tensor`) → `error.line` = deepest `<deepfriend>` frame line.
- `analyze` merge rule (implemented in Task 7, tested here via helper `merge(static_models, runtime) -> models`): runtime int shapes override static hints line-by-line; static `?` lines become known if runtime has them; runtime error with no static Diag → Diag `runtime_error {message}`; params from runtime override; `verified=True`.
- [ ] **Step 1: Failing tests**: user file with `train()` loop and `print` at top level + `open("x")` → only class code runs (assert no file created); settrace lines equal real-execution shapes on 10 fixture models (compare to running the same model on CPU with real zeros); custom op `x = x * self.scale` (static unsupported) becomes known; `test_runtime_timeout_falls_back` (Review Focus 4): `__init__` with `while True: pass` → `check` returns None in ≤3.5 s and next call still works; `torch.randn(10**10)` under meta succeeds instantly (meta) — assert returns result.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** commit `feat(py): torch meta runtime check`.

### Task 7: `analyze` orchestration

**Files:** Create `python/deepfriend/shapes/analyze.py`; Modify `python/deepfriend/server.py`; Test `python/tests/test_analyze.py`.

**Interfaces:** Produces `analyze.Analyzer` with `analyze(uri: str, text: str, runtime: bool) -> AnalysisResult dict`. Detects framework by imports (`torch` → pytorch, `tensorflow`/`keras` → tensorflow; both → pytorch for models found by torch analyzer, keras models still reported). Caches by `sha1(text)`. On `SyntaxError` returns the previous successful result for that uri with `syntaxError` set (Review Focus 1). UTF-16 column conversion helper `to_utf16_col(line_text, py_col)`. Server registers `analyze`; requests are processed by a worker thread with a single-slot queue so a newer `analyze` for the same uri drops an older pending one (dropped request responds `{"id":n,"result":null,"cancelled":true}`).
- [ ] **Step 1: Failing tests**: `test_syntax_error_keeps_previous`; `test_korean_identifiers_utf16_cols` (column of `10000` after a Korean comment on same line — cols in UTF-16); `test_perf_static_1000_lines` (generated file with 30 models, analyze ≤100 ms median of 5 runs, runtime=False); `test_runtime_merge_marks_verified`; `test_latest_wins_cancels_older` via server subprocess.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** commit `feat(py): analyze orchestration`.

### Task 8: Extension core — i18n, Python finder, server client, status bar, terminal env

**Files:** Create `extension/src/i18n/{i18n.ts,ko.ts,en.ts}`, `python/findPython.ts`, `server/{protocol.ts,client.ts,types.ts}`, `status.ts`, `config.ts`, `terminalEnv.ts`, `report.ts`; Modify `extension.ts`, `package.json` (settings, commands); Test `test/unit/{i18n,findPython,protocol,client}.test.ts`, `test/suite/server.test.ts`.

**Interfaces:**
- `t(key: StringKey, args?: Record<string, string|number>): string` — reads `deepfriend.language`; `{name}` placeholders. `ko.ts` is the source of `StringKey` (`export const ko = {...} as const; type StringKey = keyof typeof ko`); `en.ts` typed `Record<StringKey,string>` so a missing key is a compile error.
- `findPython(opts: {setting?: string; pythonExtPath?: string; platform: NodeJS.Platform; which: (cmd)=>Promise<string|null>; probe: (cmd: string[])=>Promise<{version:string}|null>}): Promise<PythonInfo|null>`; `PythonInfo = {command: string[]; version: string; source: "setting"|"pythonExtension"|"path"}`. Order per spec 4.1; win32 candidates `py -3`, `python`, `python3`; rejects `<3.9`; rejects Windows Store stub (`WindowsApps\python.exe` that prints nothing).
- `AnalysisClient`: `start(python: PythonInfo, bundledPath: string)`, `request<T>(method, params, token?) : Promise<T>`, `onStatus: Event<"starting"|"ready"|"busy"|"error"|"noPython">`, auto-restart with backoff 0.5 s·2ⁿ (max 8 s), give up after 5 crashes in 180 s; `dispose()`. Spawn env: `PYTHONPATH=<bundled>` (+ existing), `PYTHONIOENCODING=utf-8`, `PYTHONUNBUFFERED=1`.
- `StatusBar.set(state, framework?)`; text `$(pulse) DeepFriend: 분석 중` / `$(check) DeepFriend: 준비됨 · PyTorch` / `$(error) DeepFriend: 오류`; click → `deepfriend.showStatus` quick pick (python path/version, torch/tf versions, open log, restart, select python).
- `terminalEnv.apply(ctx)`: `ctx.environmentVariableCollection.prepend("PYTHONPATH", bundled + path.delimiter)`, `replace("DEEPFRIEND_WORKSPACE", folder)`; description localized.
- `report.openIssue(error?: Error)`: opens `https://github.com/mrpark10/DeepFriend/issues/new?title=…&body=…` with version/OS/VS Code/Python/stack, no user code.
- Settings contributed exactly as spec 6.2 (`deepfriend.language`, `.level`, `.ai.provider`, `.ai.consent`, `.ai.anthropicModel`, `.ai.openaiBaseUrl`, `.ai.openaiModel`, `.autoRecord`, `.pythonPath`, `.shapeHints.enabled`, `.runtimeCheck`, `.errorReporting.prompt`), titles via `%…%` in `package.nls.json` (ko) / `package.nls.en.json`.
- [ ] **Step 1: Failing unit tests**: `t` switches language and substitutes args; every `en` value non-empty (`i18n.test.ts`); `findPython` cases: setting wins, python-ext next, PATH fallback, rejects 3.8, win32 `py -3` candidate with path `C:\Users\박준호\AppData\…\python.exe` (Review Focus 3), store stub rejected; protocol line splitter handles chunks split mid-UTF-8 multibyte char; client restarts after kill and gives up after 5 crashes (fake process).
- [ ] **Step 2: Integration test** `server.test.ts`: real activation in test-electron with `deepfriend.pythonPath` = `.venv/bin/python`, status reaches `ready`, `initialize` reports torch version.
- [ ] **Step 3–4:** FAIL → implement → PASS. **Step 5:** commit `feat(ext): core client, i18n, python finder`.

### Task 9: F1 UI — hints, diagnostics, quick fix, CodeLens

**Files:** Create `extension/src/f1/*.ts`; Test `test/unit/analysisStore.test.ts`, `test/suite/f1.test.ts`, fixtures `test/fixtures/f1/*.py`.

**Interfaces:**
- `AnalysisStore`: `get(doc): AnalysisResult|undefined`, `onDidChange: Event<Uri>`; schedules analyze on open, on save immediately, on change after 400 ms idle; only for `languageId==="python"` docs whose text matches `/\b(import|from)\s+(torch|tensorflow|keras)\b/`; sends `runtime: config.runtimeCheck`; ignores cancelled/stale responses (document version check).
- `InlayHintsProvider` → for each Hint an `InlayHint` at end of line, label `→ ${shape}`, `paddingLeft`, tooltip markdown (all shapes / unknown reason localized).
- Diagnostics collection `deepfriend` severity Error, source `DeepFriend`, message `t("diag."+code, args)`; ko copy for `linear_in` exactly: `앞 레이어 출력은 {actual}개인데 Linear는 {expected}개를 기대해요. in_features를 {actual}로 바꿀까요?`
- `CodeActionProvider` (QuickFix, `isPreferred`) title ko `in_features를 {actual}로 바꾸기` (per code: `in_channels`, `num_features`).
- `CodeLensProvider` above model class when `needsInput`: `📐 입력 크기 정하기` → `deepfriend.setInputShape` → InputBox prefilled with `suggestedInput ?? "(B, 3, 32, 32)"`, validated by regex, inserts `# df: input …` line above class with the class's indentation.
- [ ] **Step 1: Failing integration tests** (`executeInlayHintProvider`, `languages.getDiagnostics`, `executeCodeActionProvider`, `executeCodeLensProvider`): spec example hint text; Linear mismatch diag message exact Korean string; applying quick fix makes diag disappear after save; CodeLens inserts comment and hints appear; English setting changes message; file without torch import → no server request (spy); timing: save → hints updated ≤1000 ms (assert).
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** commit `feat(ext): F1 shape hints`.

### Task 10: F7 concept hover

**Files:** Create `extension/src/f7/{concepts.ts,hover.ts}`; Test `test/unit/concepts.test.ts`, `test/suite/f7.test.ts`.

**Interfaces:** `concepts.ts` exports `CONCEPTS: Record<string, Concept>` keyed by canonical name (`torch.nn.Conv2d`, `keras.layers.Dense`, …) with `Concept = {title: Lang2; what: {beginner: Lang2; normal: Lang2}; diagram?: string; params: Record<string, Lang2>; doc: string}` where `Lang2 = {ko: string; en: string}`. Coverage list: torch Linear, Conv1d, Conv2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Flatten, BatchNorm1d, BatchNorm2d, LayerNorm, Dropout, ReLU, LeakyReLU, GELU, Sigmoid, Tanh, Softmax, LogSoftmax, Embedding, LSTM, GRU, RNN, Sequential, ModuleList, CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss, optim.SGD, optim.Adam, optim.AdamW, DataLoader; keras Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Embedding, LSTM, Input, Sequential. `resolveName(lineText, col) -> {concept, param?}` handles `nn.X`, `torch.nn.X`, `layers.X`, `keras.layers.X`, `tf.keras.layers.X`, bare imported names, and keyword args inside the enclosing layer call. Hover = title, level text, diagram in code block, params (all, or the hovered one bold first), F1 shape if a hint exists on that line, doc link.
- [ ] **Step 1: Failing tests**: every concept has non-empty ko/en for both levels and every param; `resolveName` cases incl. hovering `kernel_size` inside multi-line `nn.Conv2d(` call; integration `executeHoverProvider` on `nn.Conv2d` returns Korean beginner text; level setting `normal` changes text.
- [ ] **Step 2–4:** FAIL → implement (write the copy carefully: plain Korean, 해요체, one paragraph) → PASS. **Step 5:** commit `feat(ext): F7 concept hover`.

### Task 11: F3 traceback parser and offline dictionary

**Files:** Create `extension/src/f3/{traceback.ts,errorDictionary.ts}`; Test `test/unit/{traceback,errorDictionary}.test.ts`, fixtures `test/fixtures/errors/*.txt` (real error outputs captured from `.venv` torch/tf).

**Interfaces:** `findTracebacks(output: string): ParsedError[]` (handles ANSI codes, CRLF, chained "During handling…" — keep last, IPython/Jupyter style `---> 12` frames, SyntaxError/IndentationError without "Traceback" header block form `File "x", line N` + caret); `ParsedError = {type: string; message: string; frames: {file: string; line: number; name: string}[]; raw: string}`; `userFrame(err, workspaceFolders): Frame|undefined` (last frame inside a workspace folder and not under `site-packages`/`dist-packages`/`lib/python`). `explain(err): Explanation|null`; `Explanation = {id: string; title: Lang2; meaning: Lang2; cause: Lang2; fix: Lang2; actions: ("installModule"|"goToLine"|"search")[]; vars: Record<string,string>}`. Entries (ids): `shape_matmul, shape_conv_channels, shape_tensor_size, shape_view, shape_target_batch, shape_keras_incompatible, shape_keras_input, cuda_oom, tf_oom, dtype_scalar, dtype_matmul, dtype_long_target, device_mismatch, device_input_weight, cuda_unavailable, module_not_found, indentation, name_error, attribute_none, attribute_error, type_args, index_error, key_error, file_not_found, zero_division` — each with regex extracting numbers into copy (e.g. `mat1 and mat2 shapes cannot be multiplied (32x12544 and 10000x10)` → "앞 레이어가 12544개를 내보내는데 다음 Linear는 10000개를 받아요").
- [ ] **Step 1:** Capture fixtures by running small scripts in `.venv` that trigger each error (script `python/tests/tools/make_error_fixtures.py`, output files committed).
- [ ] **Step 2: Failing tests**: each fixture maps to its id with extracted vars; `parses windows traceback with korean path` (Review Focus 3) `File "C:\Users\박준호\딥러닝 과제\train.py", line 12`; Jupyter-style output; chained exceptions pick last; unknown error → null.
- [ ] **Step 3–4:** FAIL → implement → PASS. **Step 5:** commit `feat(ext): F3 parser and dictionary`.

### Task 12: F3 watchers and error card

**Files:** Create `f3/{terminalWatcher.ts,notebookWatcher.ts,errorCard.ts}`, `webview/html.ts`, `media/webview/{common.css,card.js}`; Test `test/suite/f3.test.ts`.

**Interfaces:** `terminalWatcher` uses `window.onDidStartTerminalShellExecution` (feature-detected) and `execution.read()`; buffers up to 200 KB per execution; on each complete traceback → `ErrorHub.report(parsed, source)`. `notebookWatcher` listens `workspace.onDidChangeNotebookDocument`, reads outputs with mime `application/vnd.code.notebook.error` (`{name, message, stack}`). `.deepfriend/runs/*/error.json` watcher feeds the same hub. `ErrorHub` dedups same `type+message+userFrame` within 10 s. Notification: `showErrorMessage(title, "자세히", "에러 난 줄로 이동")` (+ `"설치"` for module_not_found → runs `<python> -m pip install <module>` in a new terminal named `DeepFriend`; module name mapping `cv2→opencv-python, sklearn→scikit-learn, PIL→pillow, yaml→pyyaml, deepfriend→deepfriend`). "자세히" → `ErrorCard` webview in `ViewColumn.Beside`, reused, showing 무슨 뜻/원인/고치는 법 + raw + buttons; unknown errors show "아직 사전에 없는 에러예요" + "AI에게 물어보기" (Task 16 hook, hidden when AI off) + "검색" (opens `https://www.google.com/search?q=<type: message>` after confirm).
  `webview/html.ts`: `renderWebview(webview, {script, title, body})` with strict CSP (`default-src 'none'; img-src ${cspSource} data:; style-src ${cspSource}; script-src 'nonce-…'`), VS Code theme CSS variables.
- [ ] **Step 1: Failing integration tests**: simulate hub report → card panel opens with Korean title; go-to-line opens file at line; notebook output event triggers hub (construct `NotebookCellOutput` via `vscode.NotebookCellOutputItem.error`); error.json file creation triggers hub; dedup works.
- [ ] **Step 2–4:** FAIL → implement → PASS. Manual check in Task 21 that real terminal run triggers it. **Step 5:** commit `feat(ext): F3 error cards`.

### Task 13: Python `watch()` collector

**Files:** Create `python/deepfriend/watch/*.py`; Modify `python/deepfriend/__init__.py`; Test `python/tests/watch/test_*.py`.

**Interfaces:** Public API in `deepfriend/__init__.py`: `watch(name: str|None = None, config: dict|None = None, record: bool|None = None) -> Run` (idempotent; second call returns same Run), `log(step: int|None = None, **metrics) -> None`, `keras_callback() -> keras.callbacks.Callback`, `stop(status="finished") -> None`, `export(model, example_input, out_dir="deepfriend_export") -> dict` (Task 19). `record=None` → env `DEEPFRIEND_RECORD` (`"0"` disables, set by extension from `deepfriend.autoRecord`) else True; when False, write to `.deepfriend/live/<id>/` and delete on exit.
- `Run` writes per contract; buffered writer thread flushes every 0.5 s and on exit (`atexit`); `meta.json` written atomically (`.tmp` + `os.replace`).
- Root discovery per spec 4.3. Creates `.deepfriend/.gitignore`.
- `torch_hooks.install(run)`: wraps `torch.Tensor.backward` (scalar losses only → `split=train`), registers `torch.nn.modules.module.register_module_forward_hook` filtered to `_Loss` subclasses: when `not torch.is_grad_enabled()` → `split=val`; CE/NLL → computes accuracy (`argmax(-1)==target`) and `numClasses = input.shape[-1]`; wraps `torch.optim.Optimizer.step` → first call records `optimizer` class name and `lr` from `param_groups[0]` (and every later lr change); wraps `DataLoader.__iter__` → `batch_size`, epoch increments when a train loader iterator is created again; first train batch → `samples.capture(batch)`. All wrappers restore originals on `stop()`.
- Epoch aggregation: on epoch change, writes an aggregated record per split (mean loss, acc over the epoch) with `"agg": true`.
- `keras_cb`: on `train_begin` reads `params` (epochs, steps) and optimizer lr; `on_train_batch_end` → train step loss/acc; `on_epoch_end` → agg records for train & val from logs. `watch()` patches `keras.Model.fit` (if `tensorflow`/`keras` in `sys.modules`) to append callback when none is a DeepFriend callback.
- `snapshot.capture(run)`: `__main__.__file__` + modules in `sys.modules` whose files are under root and not under `.venv`/`site-packages`, ≤5 MB total, at `watch()` time and again at exit (hash-checked).
- `excepthook.install(run)`: chains previous `sys.excepthook`; writes `error.json`; KeyboardInterrupt → status `stopped`; other → `failed`. Also `threading.excepthook`. IPython: registers `set_custom_exc` when available.
- Git hash via `git rev-parse HEAD` (timeout 1 s, ignore failure).
- [ ] **Step 1: Failing tests** (subprocess scripts run with `.venv`, tmp project dir): torch tiny MLP 2 epochs on random data with val loop under `no_grad` → metrics has train steps, val agg per epoch, acc, `numClasses==3`, `config.lr==0.01`, `batch_size==16`, `optimizer=="Adam"`, samples.json has ≤8 items, `code/train.py` exists, status `finished`; Keras tiny model with `watch()` only (no explicit callback) → val_loss recorded; `test_nan_loss_written_as_null_flag`; `test_keyboard_interrupt_marks_stopped` (Review Focus 5: send SIGINT); `test_exception_writes_error_json`; `test_watch_never_raises` (monkeypatch writer to raise → training completes); `test_record_false_cleans_up`; overhead test: 2000 tiny steps with watch ≤ 1.3× without.
- [ ] **Step 2–4:** FAIL → implement → PASS (also `.venv39` for the stdlib-only import test: `import deepfriend` without torch works and `watch()` writes meta). **Step 5:** commit `feat(py): watch collector`.

### Task 14: F2 doctor rules + calibration examples

**Files:** Create `extension/src/f2/doctorRules.ts`, `examples/05_doctor/{nan,overfit,plateau,oscillation,chance,healthy}.py`; Test `test/unit/doctorRules.test.ts`, fixtures `test/fixtures/runs/<case>/` (real outputs from running the examples).

**Interfaces:** `diagnose(series: RunSeries): Finding[]`; `RunSeries = {train: {step:number; loss:number|null; nan?:boolean; acc?:number}[]; epochs: {epoch:number; loss?:number; val_loss?:number; acc?:number; val_acc?:number}[]; numClasses: number|null}`; `Finding = {id: "nan"|"overfit"|"plateau"|"oscillation"|"chance"; args: Record<string,number>; actions: DoctorAction[]}`; `DoctorAction = {kind:"setLr"; factor:number} | {kind:"setBatch"; factor:number} | {kind:"snippet"; id:"earlyStopping"|"dropout"|"normalize"|"eps"} | {kind:"showSamples"}`. Starting thresholds (tune only with fixtures, record final values in code constants with comments): nan = any; overfit per spec doc table (≥3 val epochs, 2 consecutive rises after min, last val > min·1.02, train at last < train at min); plateau = ≥50 train points, first-10% vs last-10% mean relative change < 1% (falls back to epoch losses if ≥3 epochs); oscillation = last ≤200 points, residual std after linear detrend / mean ≥ 0.25 and relative decrease over window < 10%; chance = numClasses ≥2 and (≥3 epochs or ≥200 steps) and best acc ≤ 1.5/C and last acc ≤ 1.2/C. Priority if several: nan > chance > overfit > oscillation > plateau (return all, sorted).
- Examples: CPU-only, synthetic data, finish in <60 s each, `import deepfriend; deepfriend.watch()`; each crafted to show exactly its problem (e.g. `nan.py`: lr=10 with SGD; `chance.py`: shuffled labels; `overfit.py`: tiny train set big MLP; `plateau.py`: lr=1e-7; `oscillation.py`: lr high + batch 2); `healthy.py`: should produce no finding.
- [ ] **Step 1:** Run each example in `.venv`, copy resulting run folders to fixtures.
- [ ] **Step 2: Failing tests**: each fixture → exactly its expected finding id first; healthy → `[]`; synthetic edge cases (empty series → [], single point → [], all-NaN → nan only, val missing → no overfit).
- [ ] **Step 3–4:** FAIL → implement/tune → PASS. **Step 5:** commit `feat: F2 doctor rules + calibration examples`.

### Task 15: F2 doctor panel, run watcher, actions

**Files:** Create `f2/{runReader.ts,runWatcher.ts,doctorView.ts,actions.ts}`, `media/webview/{doctor.js,samples.js}`, vendor uPlot copy step in build; Modify `python/deepfriend/server.py` + Create `python/deepfriend/findargs.py`; Test `test/unit/runReader.test.ts`, `python/tests/test_findargs.py`, `test/suite/f2.test.ts`.

**Interfaces:**
- `runReader.readRun(dir): Promise<RunData>` (`RunData = {meta; series: RunSeries; samples?}`), incremental `tailMetrics(dir, offset) -> {records, offset}` that ignores a truncated last line (Review Focus 5) and NaN flags.
- `RunWatcher`: `FileSystemWatcher` on `**/.deepfriend/{runs,live}/*/{meta.json,metrics.jsonl}` + 1 s poll fallback; `onDidUpdate: Event<RunData>`; current run = most recent `running`; marks `running` runs whose pid is dead as `stopped` (in memory, and writes meta); if `meta.deepfriendVersion` is lower than the bundled version (i.e. an old pip-installed copy was used, e.g. in a notebook), shows once per session `t("update.pythonPackage")` with button "업데이트" → terminal `<python> -m pip install -U deepfriend` (spec 운영 준비: 업데이트).
- `DoctorView` (`WebviewViewProvider` id `deepfriend.doctor` in panel container `deepfriendPanel`, title "학습 닥터"): left = uPlot loss (train/val) + accuracy charts updated each second (decimate to ≤2000 points); right = finding cards (3 lines 무슨 일이에요/왜 그래요/이렇게 해보세요 + action buttons). Empty state shows the one-liner `import deepfriend; deepfriend.watch()` with copy button.
- `findargs.find_argument(text, kind) -> list[ArgLocation]` (server method `findArgument`): `lr` in calls to `optim.<SGD|Adam|AdamW|RMSprop|Adagrad|Adadelta|Adamax|NAdam|RAdam>` (any prefix) and `learning_rate` in keras optimizers (`keras.optimizers.X`, `optimizers.X`, `tf.keras.optimizers.X`), `batch_size` in `DataLoader(...)` and `.fit(...)`; missing arg → `insert: true` with `defaultValue` (Adam/AdamW/Adamax/NAdam/RAdam 1e-3, SGD 1e-3, RMSprop 1e-2, Adagrad 1e-2, Adadelta 1.0, keras optimizers 1e-3; DataLoader batch 1, fit batch 32).
- `actions.apply(action, run)`: opens `run.meta.script`, calls `findArgument`, computes new value (`value·factor`, formatted as `1e-4` style when <0.001), shows `vscode.diff` preview of a temp copy, then modal "적용할까요?" → WorkspaceEdit. Snippet actions open an untitled python doc with commented snippet. `showSamples` opens samples webview (grid of images/vectors with labels).
- [ ] **Step 1: Failing tests**: tail ignores truncated line; findargs exact spans for `optim.Adam(model.parameters(), lr=0.01)`, insertion for `optim.Adam(model.parameters())`, keras `Adam(learning_rate=1e-3)`, `DataLoader(ds, batch_size=2, shuffle=True)`; integration: writing a fixture run into workspace `.deepfriend/runs` makes `DoctorView` post a `findings` message containing `"nan"` (test hook `getLastPostedMessage()`); applying setLr on a fixture script changes `lr=10` → `lr=1` after confirm (stub modal via test override).
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** commit `feat: F2 doctor panel`.

### Task 16: AI module (optional wording)

**Files:** Create `extension/src/ai/{ai.ts,consent.ts,providers.ts}`; Modify `f2/doctorView.ts`, `f3/errorCard.ts`; Test `test/unit/ai.test.ts`.

**Interfaces:** Read the `claude-api` skill before writing the Anthropic provider (model id and request shape). `polish(kind: "doctor"|"error", payload: {facts: string; code?: string}, token): Promise<string|null>`; `askUnknownError(err: ParsedError, codeLines: string): Promise<Explanation|null>`. Providers: `vscode` (`vscode.lm.selectChatModels()` first available; feature-detect), `anthropic` (fetch `https://api.anthropic.com/v1/messages`, key from `SecretStorage` key `deepfriend.anthropicKey`, default model from `deepfriend.ai.anthropicModel`), `openai-compatible` (`${baseUrl}/chat/completions`, key `deepfriend.openaiKey`). System prompt fixed: easy Korean (or English), 해요체, ≤3 sentences per section, must not invent numbers not in facts, output JSON `{meaning, cause, fix}` for errors. `consent.ensure(preview: string): Promise<boolean>` shows modal with exactly what will be sent; buttons 이번만 허용 / 항상 허용 (sets `ai.consent=always`) / 사용 안 함 (sets `never`). Code sent ≤10 lines around user frame. Results rendered with badge `AI 제안`. Timeout 15 s; any failure → null (rule text stays).
- [ ] **Step 1: Failing tests** (providers injected as fakes): consent `never` → provider not called; `ask` + deny → null; payload code truncated to 10 lines; invalid JSON from model → null; `off` provider → null without prompting.
- [ ] **Step 2–4:** FAIL → implement → PASS; manual check with `vscode` provider in Task 21 if Copilot is available on this machine. **Step 5:** commit `feat: optional AI wording`.

### Task 17: F4 model structure view

**Files:** Create `f4/modelView.ts`, `media/webview/model.js`; Modify `package.json` (activitybar container `deepfriend` with icon `media/deepfriend.svg`, views `deepfriend.model` (webview) and `deepfriend.runs`); Test `test/suite/f4.test.ts`.

**Interfaces:** `ModelView` (`WebviewViewProvider` `deepfriend.model`) subscribes to `AnalysisStore.onDidChange` and `window.onDidChangeTextEditorSelection`; picks model whose `[line,endLine]` contains cursor else first; posts `{type:"model", model: ModelInfo, activeLine}`; webview draws vertical blocks (name, kind, outShape, params formatted `1.2K/3.4M`), red border for `error`, grey for `?`, header totals + memory with breakdown tooltip; click → `{type:"reveal", line}` → editor reveal + select line; active block highlighted. Command `deepfriend.openModelView` focuses the view.
- [ ] **Step 1: Failing integration tests**: opening fixture LeNet posts model with 7 blocks and totalParams exact; moving cursor to another class switches model; reveal message moves editor selection to line.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** commit `feat: F4 model view`.

### Task 18: F5 experiments tree, compare, restore

**Files:** Create `f5/{runsTree.ts,compare.ts,restore.ts}`, `media/webview/compare.js`; Test `test/unit/compare.test.ts`, `test/suite/f5.test.ts`.

**Interfaces:** `RunsTreeProvider` (`deepfriend.runs`, `canSelectMany: true`): items sorted newest first, label = `meta.name ?? script basename` + time, description = final metric (`val_acc` → `acc` → `val_loss` → `loss`), icon star/running/failed/stopped; context menu commands `deepfriend.openRun`, `deepfriend.toggleStar` (writes `starred` atomically), `deepfriend.restoreRun`, `deepfriend.deleteRun` (confirm, moves to OS trash via `workspace.fs.delete({useTrash:true})`), `deepfriend.compareRuns` (needs exactly 2 selected; else quick pick of runs). `compare.configDiff(a, b): {key, a, b, same}[]` (union of config + model + numClasses keys, sorted, differing first). Compare panel: diff table, overlaid charts (same uPlot code as doctor), per-file "코드 비교" buttons → `vscode.diff(a/code/f, b/code/f)`. `restore.restore(run)`: lists files, modal confirm, backs up current versions to `.deepfriend/backups/<timestamp>/`, writes snapshot files. `openRun` shows that run in Doctor panel (pinned run).
- [ ] **Step 1: Failing tests**: configDiff ordering & values; tree shows 3 fixture runs in order with star; toggleStar persists; restore writes backup then file content equals snapshot; delete removes folder.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** commit `feat: F5 experiments`.

### Task 19: F6 export to app

**Files:** Create `python/deepfriend/export/{cli.py,convert.py,verify.py,flutter.py}`, `extension/src/f6/exportCommand.ts`; Test `python/tests/export/test_*.py`, `test/suite/f6.test.ts`.

**Interfaces:**
- First step of the task: re-check litert-torch install command, Python/torch version requirements and Windows support (web); record findings as a comment at top of `convert.py` and adjust the platform rule below.
- `python -m deepfriend.export.cli --keras <path> | --torch-file <py> --class <Name> [--weights <pt>] --input "1,3,224,224" --out <dir>`; prints progress lines and a final line `DEEPFRIEND_RESULT <json>` where json = `{ok, tflite, flutterDir, inputShape, outputShape, verify:{maxAbsDiff, cosine, passed}, error?: {code, message}}`; error codes `converter_missing`, `platform_unsupported`, `convert_failed`, `load_failed`.
- `convert.keras_to_tflite(path, out) -> str`; `convert.torch_to_tflite(model, example) -> str` (litert-torch `convert(model.eval(), (example,)).export(path)`); `verify.compare(run_original, tflite_path, example) -> dict` (`ai_edge_litert.interpreter.Interpreter`, fallback `tf.lite.Interpreter`; passed iff cosine ≥ 0.999 and maxAbsDiff ≤ 1e-3·max(1, |ref|max)). `flutter.generate(name, input_shape, output_shape) -> dict[str, str]` files: `lib/deepfriend_model.dart` (class `DeepFriendModel` with `static Future<DeepFriendModel> load()`, `List<double> predict(List<double> input)` using `tflite_flutter` `Interpreter.fromAsset('assets/<name>.tflite')`, reshape to the exact input shape, output buffer exact shape), `pubspec_snippet.yaml` (`tflite_flutter: ^0.11.0` — verify current version in step 1 — and assets entry), `README.md` (ko) steps.
- `deepfriend.export(model, example_input, out_dir)` in Python API calls the same functions.
- Extension command `deepfriend.exportToApp`: quick picks (framework → file pickers → input prefilled from F1 model input or `input_shape`), converter check `python -c "import litert_torch"`; if missing → modal offering to create `.deepfriend/export-env` (`python -m venv --system-site-packages`) and `pip install litert-torch ai-edge-litert` with progress (output channel), then run cli with that env's python; result notification with "폴더 열기", "검증 결과 보기"; `platform_unsupported` → generate `deepfriend_export/convert_in_colab.ipynb` and say so.
- [ ] **Step 1: Failing tests**: flutter generator golden files for shape `(1,28,28,1)→(1,10)`; Keras export end-to-end in `.venv` (tiny model) produces `.tflite` and `verify.passed`; torch export end-to-end (skip if litert-torch can't install on this platform, and record that); cli error JSON when converter missing (simulate via env var `DEEPFRIEND_FORCE_MISSING=1`); integration: command with stubbed picks runs cli and shows result notification.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** commit `feat: F6 export to app`.

### Task 20: Examples, walkthrough, onboarding polish

**Files:** Create `examples/01_mnist_mlp.py`, `02_cifar_cnn.py`, `03_keras_cnn.py`, `04_shape_error.py`, `extension/walkthrough/*.md`; Modify `package.json` (walkthrough `deepfriend.start` with steps: 예제 열기 → 크기 보기 → 학습 닥터 → 실험 비교 → 앱 내보내기; commands `deepfriend.openExample`), `extension.ts` (first-activation: open walkthrough once, stored in `globalState`); Test `test/suite/examples.test.ts`.

**Interfaces:** `deepfriend.openExample` → quick pick of 5 examples → copies into a chosen folder (default `~/DeepFriend-examples`) and opens it. Examples use synthetic data (no downloads) so they run offline in <60 s, each with `# df: input` comments (except 04 which demonstrates the CodeLens and a deliberate Linear mismatch).
- [ ] **Step 1: Failing tests**: each example analyzed with zero diagnostics except `04_shape_error.py` (exactly one `linear_in`); each example runs to completion in `.venv` with exit 0 (python test `python/tests/test_examples.py`, marked slow).
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** commit `feat: examples and walkthrough`.

### Task 21: Full QA pass (Claude does this by hand, plus tooling)

**Files:** Create `extension/scripts/render-webviews.mjs` (renders each webview HTML with fixture messages in headless Chromium via `playwright`, light+dark theme vars, widths 300/800 px, saves PNGs to `qa/screens/`), `qa/QA-REPORT.md`.

- [ ] **Step 1:** Run all automated suites: `pytest python` (.venv & .venv39), `npm run test:unit`, `npm run test:int`. All PASS.
- [ ] **Step 2:** Render webviews, open every PNG and check: no clipped text, readable in both themes, Korean wraps, empty states. Fix issues, re-render.
- [ ] **Step 3:** `npm run package` → install VSIX into a clean profile: `"/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" --profile DF-QA --install-extension deepfriend-0.1.0.vsix`. Walk the full journey with each example (open → hints → CodeLens → quick fix → run in integrated terminal → doctor graph/cards → errors in terminal → card → go to line → experiments compare → restore → export). Capture screenshots (macOS `screencapture`, ask user for screen-recording permission if needed) for README.
- [ ] **Step 4:** Edge checks from Review Focus 1–5 by hand; Python missing simulation (`deepfriend.pythonPath` set to bogus) shows "파이썬 필요"; kill server process (`kill -9`) → auto restart; VSIX size check (< 5 MB).
- [ ] **Step 5:** Record results, bugs found and fixes in `qa/QA-REPORT.md`; every bug fixed with a regression test. Commit `test: full QA pass`.

### Task 22: Docs, legal, marketplace listing

**Files:** Create `README.md` (ko, primary — also copied as extension README), `README.en.md`, `CHANGELOG.md`, `LICENSE` (MIT, © 2026 박준호), `docs/{guide.md,faq.md,privacy.md,privacy.en.md,terms.md,terms.en.md}`, `extension/media/icon.png` (128×128, generated from an SVG drawn for DeepFriend), `python/README.md`; Modify `extension/package.json` (`displayName`, `description` ko, `categories: ["Machine Learning","Education","Linters","Visualization"]`, `keywords` [pytorch, tensorflow, keras, deep learning, 딥러닝, 한국어, korean, education, shape, tensor], `icon`, `galleryBanner`, `repository`, `bugs`, `homepage`, `license`, `qna: "marketplace"`? → no, `bugs.url` issues).

- [ ] **Step 1:** Write docs. Privacy policy states: no data collected by the developer; all records local in `.deepfriend/`; AI (optional) sends error text + ≤10 code lines to the provider the user chose, only after consent; contact sk92977852@gmail.com; effective date 2026-10-26. Terms: free, as-is, MIT.
- [ ] **Step 2:** README: one-screen intro, GIF/screenshots per feature (from Task 21), install (just click), quick start (3 steps), support matrix, F1 supported layers list, FAQ link, privacy/terms links, English section link.
- [ ] **Step 3:** `vsce ls` shows only intended files; `vsce package` has no warnings. Commit `docs: readme, legal, listing`.

### Task 23: CI, release automation, private repo

**Files:** Create `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `docs/publishing.md`.

- [ ] **Step 1:** `ci.yml`: matrix os [ubuntu-latest, windows-latest, macos-latest] × python ["3.9","3.12"]; jobs: pytest without frameworks; pytest with torch (CPU, versions `2.2.*` and latest) and tensorflow latest on python 3.12 only (tf 3.9 wheels too); `npm ci && npm run build && npm run test:unit`; integration tests via `xvfb-run` on Linux, direct on mac/win; package VSIX artifact.
- [ ] **Step 2:** `release.yml` on tag `v*`: run CI, then `vsce publish -p ${{ secrets.VSCE_PAT }}`, `ovsx publish -p ${{ secrets.OVSX_PAT }}`, PyPI via `pypa/gh-action-pypi-publish` with trusted publishing (`id-token: write`), GitHub Release with VSIX.
- [ ] **Step 3:** `gh repo create mrpark10/DeepFriend --private --source . --push`; watch CI with `gh run watch`; fix until green on all 3 OS.
- [ ] **Step 4:** `docs/publishing.md` (ko): exact click-by-click for creating Marketplace publisher `mrpark10` + Azure DevOps PAT (scope Marketplace › Manage), Open VSX account + namespace `mrpark10` + token, PyPI account + trusted publisher (owner mrpark10, repo DeepFriend, workflow release.yml, environment pypi), adding secrets with `gh secret set`.
- [ ] **Step 5:** Commit `ci: test matrix and release pipeline`.

### Task 24: Launch

- [ ] **Step 1:** Run the spec's 출시 체크리스트 and tick each item in `qa/QA-REPORT.md` with evidence (F1–F7 on 5 examples; 3 OS CI green; auto restart; listing assets; privacy/terms; issues channel; AI method implemented).
- [ ] **Step 2:** Ask the user to: create the three publisher accounts per `docs/publishing.md`, set secrets, and confirm making the repo public (`gh repo edit --visibility public --accept-visibility-change-consequences`) — Issues must be reachable before listing goes live.
- [ ] **Step 3:** After confirmation: `git tag v0.1.0 && git push --tags`; `gh run watch` release; verify listing pages load (`curl` marketplace/open-vsx/pypi URLs) and `code --install-extension mrpark10.deepfriend` works in a fresh profile.
- [ ] **Step 4:** Commit final checklist; report to user with links.
