# DeepFriend(딥프렌드) — VS Code 확장 설계 문서

- 날짜: 2026-10-26
- 원본 명세: `AI 학습 도우미 VS Code 확장 기능명세서.pdf` (@박준호, 2026-10-26) — 이 문서는 그 명세를 실제로 만들 수 있게 구체화한 것. 명세와 다르게 정한 부분은 **[변경]**, 제가 정한 부분은 **[Claude 판단]**으로 표시.

---

## 1. 의도와 이유

DeepFriend는 VS Code에서 PyTorch·TensorFlow(Keras) 코드를 쓰는 입문~중급 학습자(고등학생·대학생)에게 **실행 전에 문제를 보여주고, 실행 후에 이유를 한국어로 설명**해 주는 확장이다. 라이브러리나 실행 환경을 대체하지 않고 그 위에 얹는 도우미다.

사용자(박준호)가 직접 정한 최우선 원칙:

1. **설치하는 사람의 불편함이 전혀 없어야 한다.** 마켓플레이스에서 설치 버튼 하나 누르면 끝나야 하고, 별도 pip 설치·API 키 발급·설정 편집을 요구하지 않아야 한다.
2. **많은 사람이 써야 한다.** 배포 범위, 첫 사용 경험, 문서, 신뢰(개인정보) 모두 이 기준으로 결정한다.
3. 이름은 **DeepFriend** (확장·파이썬 패키지 모두). 명세의 "딥메이트(가칭)"는 PyPI에 남의 `deepmate` 패키지가 이미 있어 쓰지 않는다. **[변경]**
4. 구현은 서브에이전트 없이 executing-plans로 하나씩 진행하고, **QA도 Claude가 직접 세세하게** 한다.

## 2. 목표 / 비목표 / 반목표

**목표**
- 명세의 F1~F7 일곱 기능 전부 정식 출시에 포함.
- 명세의 비기능 요구사항(속도·가벼움·개인정보·오프라인·지원 환경·안정성·언어) 충족.
- 명세의 정식 출시 체크리스트 전부 완료 후 VS Code Marketplace + Open VSX + PyPI 동시 배포.

**비목표 (이번 출시에 없음)**
- 유료 구독·로그인·결제·AI 중계 서버. (명세의 "AI 비용 방식" 중 서버 없는 쪽으로 결정 — 6.3 참고)
- F4의 "블록 드래그로 코드 생성" 양방향 편집 (명세에서도 "나중 확장").
- DataLoader를 보고 입력 크기 완전 자동 추측 (명세에서도 2차 버전). 단, 흔한 데이터셋 이름 기반의 "추천값 미리 채우기"는 넣는다.
- 원격(SSH/WSL/Codespaces) 전용 최적화. 기본적으로 동작은 하되 별도 QA 대상은 아님.

**반목표 (기술적으로 돌아가도 이러면 실패)**
- 설치 직후 사용자가 터미널에서 무언가 직접 설치해야 하는 상태.
- 크기 표시가 **틀린 숫자**를 자신 있게 보여주는 것. 모르면 반드시 `?`로 표시한다. (명세: "실제 실행 결과와 100% 같음")
- 사용자 동의 없이 코드·에러가 외부로 나가는 것.
- 사용자 코드를 동의 없이 자동 수정하는 것. 모든 수정은 버튼 + 미리보기.
- 확장 때문에 VS Code가 느려지거나, 분석기가 죽었을 때 VS Code가 영향을 받는 것.
- 사용자 학습 스크립트의 최상위 코드(학습 루프 등)를 저장할 때마다 실행해 버리는 것.

## 3. 제약

- 지원: Windows·macOS·Linux, Python **3.9 이상**(명세는 3.10+, 더 넓게 지원 **[변경/Claude 판단]** — 이 맥의 기본 파이썬이 3.9이고, 학생 PC에 오래된 파이썬이 흔함), PyTorch 2.x, TensorFlow 2.x.
- VS Code 엔진 `^1.93` (터미널 출력 읽기 API가 정식이 된 버전). Cursor·Windsurf 등 VS Code 계열 에디터에서도 설치 가능해야 함.
- 파이썬 쪽 DeepFriend 코드는 **표준 라이브러리만** 사용 (torch/tensorflow는 사용자가 설치한 것이 있으면 선택적으로 사용). 그래야 확장에 함께 넣어 pip 없이 돌 수 있음.
- 모든 문구 한국어 기본, 설정으로 영어 전환.
- 게시자 계정(VS Code Marketplace, Open VSX, PyPI)은 사용자 본인 명의로 만들어야 함 — Claude는 절차를 안내하고 나머지를 자동화.

## 4. 전체 구조

명세의 "확장은 화면, 파이썬 서버는 계산" 구조를 유지한다.

```
DeepFriend/
  extension/        TypeScript VS Code 확장 (esbuild로 번들)
    python/         ← 빌드 시 아래 python/deepfriend 를 복사해 VSIX에 동봉
  python/deepfriend/  파이썬 패키지 (표준 라이브러리만)
    server.py       표준입출력 JSON 메시지 분석 서버
    shapes/         F1 정적 크기 계산(AST) + torch 런타임 검증
    doctor.py       F2 진단 규칙
    watch.py        F2/F5 수집기: watch(), log(), Keras 콜백
    runs.py         F5 실험 기록 저장
    export/         F6 변환·검증·Flutter 코드 생성
    errors.py       F3용 excepthook 기록
  examples/         예제 모델 5개 + 학습 문제 재현 예제
  docs/             사용법·FAQ·개인정보처리방침·이용약관
  .github/workflows CI(3 OS × 버전) + 릴리스 자동 배포
```

### 4.1 설치 불편함 제로를 위한 핵심 결정 [Claude 판단]

- **분석 서버를 확장 안에 동봉.** 확장이 사용자의 파이썬으로 `python -u -m deepfriend.server`를 실행하고 `PYTHONPATH`에 동봉 경로를 넣는다. F1·F4는 pip 설치 없이 동작.
- **파이썬 자동 탐색 순서:** ① 사용자가 설정에 직접 지정한 경로 → ② Microsoft Python 확장이 선택한 인터프리터(`@vscode/python-extension` API, 인터프리터 변경 시 자동 재시작) → ③ PATH의 `python3`/`python`/(Windows) `py -3`. 버전 3.9 미만이거나 없으면 상태 표시줄에 "파이썬 필요"와 설치 안내 버튼. 파이썬이 없어도 F3(터미널 에러 해설)·F7(호버)은 확장 쪽 데이터로 동작.
- **`import deepfriend`도 설치 없이:** VS Code의 터미널 환경변수 기능으로 통합 터미널에 `PYTHONPATH`(동봉 경로)와 `DEEPFRIEND_WORKSPACE`를 자동 주입. 학습 코드에 `import deepfriend; deepfriend.watch()`만 쓰면 됨.
- **Jupyter 노트북·외부 터미널용:** 같은 패키지를 PyPI `deepfriend`로도 배포. 노트북에서 `No module named 'deepfriend'`가 뜨면 F3 카드에 "설치" 버튼(사용자가 누르면 그 인터프리터에 `pip install deepfriend`를 터미널에서 실행). 명세의 "처음 켜면 자동 pip 설치(동의 후)"는 동봉 방식 덕분에 기본 경로에서는 필요 없음 **[변경]**.
- **버전 안내:** pip로 설치된 deepfriend가 동봉본보다 낮으면 "업데이트" 버튼 안내(명세의 업데이트 항목).
- **무거운 일은 필요할 때만:** 확장은 파이썬 파일/노트북을 열 때 켜지고, 분석 서버는 torch/tensorflow/keras를 import하는 파일이 처음 열릴 때 시작.
- **첫 사용 안내:** VS Code "시작하기(Walkthrough)" 페이지 + "예제 열기" 명령(예제 파일을 새 창에 만들어 줌).

### 4.2 확장 ↔ 서버 통신

- 표준 입출력으로 줄 단위 JSON 메시지(요청 id, 메서드, 인자 / 결과 또는 에러). 명세 그대로.
- 서버가 죽으면 자동 재시작(지수 백오프, 3분 안에 5번 넘게 죽으면 멈추고 상태 표시줄 "오류" + "다시 시작"/"오류 신고" 버튼). VS Code 프로세스와 분리되어 있으므로 VS Code는 영향 없음.
- torch 검증은 서버 안의 **별도 작업 프로세스**에서 시간 제한(기본 3초)으로 실행. 멈추거나 죽으면 그 결과만 버리고 정적 결과를 유지.

### 4.3 학습 수치 수집 방식 **[변경]**

명세는 "로컬 소켓"이었지만, **프로젝트의 `.deepfriend/runs/<실행ID>/` 폴더에 파일로 기록하고 확장이 파일 변경을 감시**하는 방식으로 바꾼다. 이유: 포트·방화벽 문제가 없고, 노트북·외부 터미널·VS Code가 꺼져 있을 때도 기록이 남으며(F5와 저장소 공유), 1초 갱신 요구를 충분히 만족. 프로젝트 루트는 `DEEPFRIEND_WORKSPACE` → 스크립트 위치에서 위로 올라가며 `.deepfriend`/`.git`/`.vscode` 탐색 → 현재 폴더 순.

`.deepfriend/` 안에는 처음 만들 때 `.gitignore`(`*`)를 자동 생성해 실수로 깃에 올라가지 않게 한다.

## 5. 기능별 설계

### F1. 실시간 크기 표시 (P0)

**보이는 모습 (명세 그대로):** 레이어 줄 끝 회색 글씨 `→ (32, 64, 112, 112)`, 에러는 빨간 물결 밑줄 + 한국어 호버 + 빠른 수정.

**언제:** 파일 저장 시 + 입력 중 400ms 쉬면(저장 안 한 내용 그대로 전송). 명세의 "저장 후 1초 안"과 비교표의 "쓰는 즉시"를 둘 다 만족 **[Claude 판단]**.

**입력 크기 주석:** 모델 클래스 바로 위 `# df: input (32, 3, 224, 224)`. 명세의 `# dm: input`도 계속 인식. `forward` 인자가 여러 개면 튜플을 쉼표로 나열. 정수 대신 `B` 같은 이름도 허용(정적 계산에서는 그대로 표시, torch 검증 때는 2로 대입). Keras는 첫 레이어의 `input_shape=`/`Input(shape=)`이 있으면 주석 없이 동작.
주석이 없으면 클래스 위에 CodeLens **"📐 입력 크기 정하기"** → 입력창(흔한 데이터셋 이름이 코드에 보이면 추천값을 미리 채움: MNIST/FashionMNIST → `(B, 1, 28, 28)`, CIFAR10/100 → `(B, 3, 32, 32)`, `Resize(224)` → `(B, 3, 224, 224)`) → 주석 자동 삽입. **[Claude 판단]**

**두 단계 계산 [Claude 판단]:**
1. **정적 계산(항상, 수십 ms):** 파이썬 AST로 `nn.Module` 상속 클래스의 `__init__`(레이어 정의, `nn.Sequential`)과 `forward`(문장 순서대로 변수별 크기 추적)를 읽고 공식으로 계산. torch가 없어도 동작. 같은 파일의 사용자 정의 모듈은 재귀적으로 계산(생성자 인자가 리터럴이면).
2. **torch 검증(torch가 설치돼 있으면, 명세의 meta 장치 방식):** 파일 전체를 실행하지 않는다. AST에서 **import문(허용 목록: torch, torchvision, math, numpy, typing, collections, functools, dataclasses), 단순 상수 대입, 클래스 정의만** 뽑아 실행 → 인자 없이 생성 가능한 모델만 `torch.device("meta")`에서 생성 → `forward` 실행 중 `sys.settrace`로 줄마다 지역 변수 텐서 크기를 기록하고, 서브모듈 forward hook으로 레이어별 출력 크기·파라미터 수 기록. 결과가 정적 계산과 다르면 **torch 결과가 이긴다**. 실패(이름 없음, 데이터 의존 연산 등)하면 조용히 정적 결과 유지.

**어디에 표시:** `forward`의 대입 줄 끝에 그 변수의 크기, `__init__`의 레이어 정의 줄(및 `nn.Sequential` 안의 각 레이어 줄) 끝에 그 레이어의 출력 크기. 한 레이어가 여러 번 호출되어 크기가 다르면 `→ (…) 외 1`과 호버로 전체. 모르면 `→ ?`(호버: "이 줄은 아직 계산할 수 없어요: 이유").

**지원 레이어 (명세 + 자주 쓰는 것 추가):**
- PyTorch: Linear, Conv1d/Conv2d, MaxPool1d/2d, AvgPool1d/2d, AdaptiveAvgPool1d/2d, AdaptiveMaxPool2d, Flatten, BatchNorm1d/2d, LayerNorm, Dropout/Dropout2d, 모든 활성화 함수(ReLU, LeakyReLU, GELU, Sigmoid, Tanh, Softmax, LogSoftmax, ELU, SiLU 등), Identity, Embedding, `nn.Sequential`, 함수형(`F.relu` 등 활성화, `F.max_pool2d`, `F.avg_pool2d`, `F.dropout`, `F.softmax`), `view`/`reshape`(-1 포함), `flatten`/`torch.flatten`, `torch.cat`, `x.size(0)`/`x.shape[0]`, `+ - * /` 요소별 연산(브로드캐스팅), `squeeze`/`unsqueeze`, `permute`/`transpose`, `mean`/`sum`(dim). 공식은 PyTorch 공식 문서 그대로(패딩 `'same'`/`'valid'`, dilation, groups, `ceil_mode`의 "마지막 창은 입력 안에서 시작" 규칙 포함).
- Keras: Dense(마지막 축에만 적용 — 4D 입력도 에러 아님), Conv2D(`padding='valid'|'same'`, strides, `data_format` 기본 channels_last), MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization, Input/InputLayer, `Sequential([...])`와 `model.add(...)`. 배치 차원은 `None`.

**에러로 잡는 것:** Linear `in_features` 불일치, Conv `in_channels` 불일치, BatchNorm `num_features` 불일치, `view`/`reshape` 원소 수 불일치, `torch.cat` 다른 축 크기 불일치, 크기가 0 이하로 줄어드는 Conv/Pool, Keras Conv2D/MaxPooling2D에 4D가 아닌 입력.
**빠른 수정:** 해당 인자를 계산된 값으로 바꾸기(예: "in_features를 12544로 바꿀까요?"). 명세의 예문 형식 그대로: "앞 레이어 출력은 12544개인데 Linear는 10000개를 기대해요."

**완료 기준(명세):** 저장 후 1초 안 표시 갱신; 지원 레이어에서 실제 실행 결과와 100% 일치 — 이는 **실제 torch/tf로 돌린 결과와 자동 비교하는 테스트**(레이어·인자 조합 수백 개, 무작위 조합 포함)로 증명한다.

### F2. 학습 닥터 (P0)

**시작:** `import deepfriend; deepfriend.watch()` 한 줄. Keras는 `model.fit(..., callbacks=[deepfriend.keras_callback()])` 또는 `watch()`만 불러도 `fit`에 자동 연결.

**`watch()`가 자동으로 모으는 것 [Claude 판단]** (사용자가 숫자를 넘겨줄 필요 없게):
- PyTorch: `Tensor.backward`를 감싸 **학습 loss**를 매 스텝 기록; 손실 함수 모듈 hook으로 `torch.no_grad()` 상태에서 계산된 loss를 **검증 loss**로 기록; CrossEntropy 계열이면 logits·정답에서 **정확도와 클래스 수**를 자동 계산; `Optimizer.step`에서 학습률·옵티마이저 이름; DataLoader에서 배치 크기·에폭 경계; 첫 배치에서 입력·라벨 샘플 최대 8개(작게 줄여 저장, F2 "샘플 확인 화면"용).
- Keras: 콜백의 logs(loss, val_loss, accuracy, val_accuracy, lr) 그대로.
- 언제든 `deepfriend.log(acc=..., val_loss=...)`로 직접 추가 가능.
- 스텝 기록은 0.5초마다 묶어서 파일에 기록(학습 속도 영향 최소화). 수집 중 어떤 에러가 나도 사용자 학습은 절대 멈추지 않음(전부 try/except).

**보이는 모습:** 아래 패널 "학습 닥터" 탭(웹뷰)에 loss·정확도 그래프가 1초마다 갱신. 차트 라이브러리는 확장에 동봉(오프라인).

**진단 규칙 (명세의 5가지, 임계값은 [Claude 판단]이며 예제로 조정):**

| 상황 | 판정 | 제안 + 버튼 |
|---|---|---|
| loss가 NaN/Inf | 한 번이라도 비유한값 | 학습률 1/10 (버튼: 옵티마이저 `lr=` 값 수정), eps 더하기 안내 |
| 과적합 | 검증 loss가 최저점 대비 2에폭 연속 상승 & 같은 구간 학습 loss 하락 | Dropout 추가·데이터 늘리기·조기 종료 (버튼: 조기 종료 코드 조각 보기) |
| loss 거의 안 움직임 | 충분한 스텝 후 앞 구간 대비 뒤 구간 평균 변화 < 1% | 학습률 올리기, 입력 0~1 정규화 (버튼: `lr=` ×10) |
| loss 크게 출렁임 | 추세 제거 후 변동이 평균 대비 큼 & 하락 추세 약함 | 학습률 낮추기·배치 키우기 (버튼: `lr=` ÷10, `batch_size=` ×2) |
| 정확도 찍기 수준 | 일정 스텝 후 정확도 ≤ 1/클래스수 × 1.2 | 라벨 섞임·마지막 층 확인 (버튼: 데이터·라벨 샘플 화면 열기) |

**결과 카드(명세 그대로):** "무슨 일이에요 / 왜 그래요 / 이렇게 해보세요" 3줄 + 코드 수정 버튼. 수정 버튼은 서버가 AST로 해당 인자 위치를 찾아 **변경 미리보기(diff)** 후 사용자가 확인해야 적용. 위치를 못 찾으면 버튼 대신 안내 문장.
기본 문장은 규칙별 한국어 템플릿. AI가 켜져 있으면 문장만 쉽게 다듬고 "AI 제안" 표시(판단은 항상 규칙).

**완료 기준(명세):** 5가지 상황을 일부러 만든 예제(CPU에서 1분 안에 끝나는 합성 데이터 스크립트)에서 모두 맞게 진단 + 정상 학습 예제에서 오진 없음.

### F3. 한국어 에러 해설 (P0)

**에러 감지 경로 [Claude 판단]:**
1. VS Code 통합 터미널 출력 읽기(셸 통합 API)로 파이썬 Traceback 감지.
2. VS Code 안 Jupyter 노트북 셀의 에러 출력 감지.
3. `watch()`가 켜진 실행은 파이썬 excepthook으로 `.deepfriend/`에도 기록(셸 통합이 안 되는 터미널 대비).

**보이는 모습:** 오른쪽 아래 알림(한 줄 제목 + "자세히" + "에러 난 줄로 이동") → "자세히"는 편집기 오른쪽에 해설 카드(웹뷰): **무슨 뜻 / 원인 / 고치는 법**. "줄로 이동"은 Traceback에서 사용자 프로젝트 파일 중 가장 마지막 프레임으로.

**오프라인 사전(확장에 내장, 정규식으로 숫자·이름 뽑아 문장에 끼움):** 명세의 5종(크기 불일치, CUDA 메모리 부족, dtype 불일치, 장치 불일치, 모듈 없음 — 모듈 없음은 "설치" 버튼)을 PyTorch·TensorFlow 메시지 변형까지 포함해 다루고, 입문자가 자주 만나는 일반 파이썬 에러(IndentationError, NameError, AttributeError, TypeError 인자 개수, IndexError, KeyError, FileNotFoundError 등)도 넣는다 **[Claude 판단]** — AI 호출을 줄이고 오프라인 범위를 넓히기 위해.
사전에 없는 에러만 AI에게(켜져 있을 때, 동의 후) 묻고, AI가 없으면 "아직 사전에 없는 에러예요" + 원문 + 검색 버튼.

### F4. 모델 구조 그림 (P1)

- 왼쪽 사이드바 "DeepFriend" 아이콘 → "모델 구조" 웹뷰. 커서가 있는 모델 클래스(없으면 파일의 첫 모델)를 위→아래 블록으로. 블록마다 레이어 이름·종류·출력 크기(F1 결과 그대로)·파라미터 수. 에러 블록은 빨간 테두리, `?`는 회색.
- 상단 요약: 전체 파라미터 수, **대략적 메모리**(파라미터 + 기울기 + Adam 상태 + 입력 배치 기준 중간값, float32 가정 — 계산식을 호버로 보여줌).
- 블록 클릭 → 해당 코드 줄로 이동. 코드 수정 → 그림 자동 갱신. 편집기 커서 이동 → 해당 블록 강조(서로 연동).
- 명령 팔레트 "DeepFriend: 모델 그림 열기"(명세의 "딥메이트: …" 명령은 모두 "DeepFriend: …"로).

### F5. 실험 자동 기록 (P1)

- `watch()`가 켜진 실행마다 `.deepfriend/runs/<날짜-시간-짧은ID>/`에: `meta.json`(설정값: 학습률·배치 크기·옵티마이저·에폭, 모델 이름·파라미터 수, 스크립트 경로, 시작/끝, 상태[실행 중/완료/실패/중단], 최종·최고 점수, 별표, deepfriend 버전, 깃 커밋이 있으면 해시), `metrics.jsonl`, `code/`(실행한 스크립트 + 프로젝트 안에서 import된 .py 파일 사본, 합계 5MB 제한).
- "코드에서 바뀐 부분" = 직전 실행 사본과의 diff (표시할 때 계산).
- 사이드바 "실험" 목록(트리): 최근순, 최종 점수, 별표 아이콘, 실행 중 표시. 우클릭/버튼: 그래프 보기, 별표, 코드 되돌리기, 삭제. 두 개 선택 → "비교".
- **비교 화면:** 설정 차이표(다른 값 강조) + 그래프 겹쳐보기 + "코드 비교"(VS Code 내장 diff 화면).
- **되돌리기:** 확인창 → 현재 파일을 `.deepfriend/backups/`에 먼저 백업 → 사본으로 덮어쓰기.
- 설정 "자동 기록 끄기"면 `watch()`는 그래프만 보여주고 디스크에 남기지 않음(학습 닥터용 임시 파일은 실행 끝나면 삭제).
- 인터넷 사용 없음(명세).

### F6. 앱으로 내보내기 (P2)

- 명령 "DeepFriend: 앱으로 내보내기" → 단계별 선택: ① 모델 선택(Keras: `.keras`/`.h5` 파일 / PyTorch: 모델 클래스 + 가중치 `.pt/.pth` 파일, 또는 학습 코드 끝에 `deepfriend.export(model, example_input)` 한 줄) ② 입력 크기(F1 결과로 미리 채움).
- 변환: Keras → TensorFlow 내장 TFLite 변환기. PyTorch → `litert-torch`(구 ai-edge-torch, 2026-10 기준 최신 공식 변환기 — 구현 시작할 때 다시 확인). 변환 도구가 없으면 "변환 도구 설치(약 N MB)" 동의 버튼 → 별도 가상환경(`.deepfriend/export-env`)에 설치해 사용자 환경을 더럽히지 않음 **[Claude 판단]**. 변환기가 지원하지 않는 OS(구현 시 확인, 예: Windows)에서는 그 사실을 알리고 Colab에서 바로 돌릴 수 있는 변환 노트북을 생성.
- **자동 확인(명세):** 같은 무작위 입력을 원본·변환본에 넣어 최대 오차·코사인 유사도 비교, 기준 통과/실패를 결과 카드로.
- **Flutter 예제(명세):** `deepfriend_export/<모델명>/`에 `model.tflite`, `flutter/lib/deepfriend_model.dart`(`tflite_flutter` 패키지로 로드·추론, 입력·출력 크기 자동 기입), `pubspec` 추가 내용, 사용법 README, 검증 리포트.
- 결과는 오른쪽 알림 카드 + "폴더 열기".

### F7. 개념 설명 호버 (P2)

- `nn.Conv2d`, `layers.Dense` 같은 이름에 마우스 → "이 레이어가 하는 일" 한 문단 + 간단한 그림(텍스트 도식) + 공식 문서 링크.
- 레이어 호출 안의 매개변수 이름(`kernel_size=`, `stride=` …)에 마우스 → 그 매개변수 한 줄 설명.
- 설정 "설명 난이도: 입문/보통"에 따라 문장이 다름. 한국어/영어.
- 데이터는 확장에 내장(오프라인, 즉시). F1에서 크기가 있는 줄이면 호버에 크기 변화도 함께.

## 6. 공통 설계

### 6.1 화면 배치 (명세 그대로)
- 편집기 안: F1 회색 글씨·밑줄·빠른 수정, F7 호버, CodeLens(입력 크기 정하기).
- 왼쪽 사이드바 "DeepFriend": F4 모델 구조, F5 실험 목록.
- 아래 패널 "학습 닥터" 탭: F2.
- 오른쪽 알림/카드: F3, F6 결과.
- 명령 팔레트: 모델 그림 열기, 실험 비교, 앱으로 내보내기, 예제 열기, 분석기 다시 시작, 파이썬 선택, AI 설정, 오류 신고.
- 상태 표시줄: `DeepFriend: 분석 중 / 준비됨 / 오류 · PyTorch|TensorFlow`. 클릭 → 상태 상세(파이썬 경로·버전·torch/tf 버전·서버 로그 열기).

### 6.2 설정
- `deepfriend.language`: `ko`(기본) / `en`
- `deepfriend.level`: `beginner`(입문, 기본) / `normal`
- `deepfriend.ai.provider`: `off` / `vscode`(VS Code에 내장된 언어 모델 — Copilot 등, 키 불필요) / `anthropic` / `openai-compatible`(OpenAI·Gemini·로컬 Ollama 등 주소 지정). 기본은 `vscode`이고 사용 가능한 모델이 없으면 조용히 규칙만 사용.
- `deepfriend.ai.consent`: `ask`(기본) / `always` / `never` — AI에 보내기 전 보낼 내용을 보여주고 물음.
- `deepfriend.autoRecord`: 켜기(기본)/끄기
- `deepfriend.pythonPath`: 비워두면 자동 탐색
- `deepfriend.shapeHints.enabled`, `deepfriend.runtimeCheck`(torch 검증, 기본 켬)
- API 키는 설정 파일이 아닌 VS Code 보안 저장소에 저장(명령 "AI 키 입력").
- 설정 제목·설명(package.nls)은 VS Code 표시 언어를 따르고, 확장 화면 문구는 `deepfriend.language`를 따른다.

### 6.3 AI 비용 방식 (명세의 "결정 필요") [Claude 판단]
명세의 두 선택지 중 **서버 없는 쪽**을 택하되, 키 발급 번거로움을 없애기 위해 **VS Code 내장 언어 모델(키 불필요)을 기본**으로 둔다. 원하는 사람만 자기 키(Anthropic 또는 OpenAI 호환)를 넣는다. 운영 비용 0원, 서버·결제 없음. AI 없이도 F1·F3 사전·F5와 F2 규칙 진단은 전부 무료로 동작.
AI에 보내는 것은 에러 메시지 + 관련 코드 최대 10줄 또는 진단 규칙 결과 + 수치 요약뿐. 보내기 전 사용자에게 내용을 보여주고 동의를 받는다(명세 개인정보 요구). AI 문장에는 항상 "AI 제안" 표시, 코드 자동 적용 없음.

### 6.4 오류 수집 [Claude 판단]
별도 수집 서버를 두지 않는다. 확장 내부 오류가 나면 "오류 신고" 버튼 → 오류 정보(코드 내용 제외, 버전·OS·스택만)를 미리 채운 GitHub Issue 작성 화면을 브라우저로 연다. 사용자가 보고 직접 제출하므로 "허락한 경우에만 전송"을 자연스럽게 만족. 설정으로 오류 신고 안내 자체를 끌 수 있음.

### 6.5 성능·가벼움
- 확장 쪽은 계산 없음, 결과 표시만. 같은 내용이면 재분석 안 함(해시 캐시), 진행 중 요청은 최신 것만 남기고 취소.
- 정적 분석 목표 100ms 이하(1000줄 파일 기준), torch 검증은 작업 프로세스를 미리 띄워 두어(torch import 1회) 보통 1초 안.
- 측정을 테스트로 고정(성능 회귀 테스트).

### 6.6 언어
- 모든 문구는 한/영 두 벌. 런타임 문구는 한 곳의 문구 사전에서 관리하고, 테스트로 "한국어에만 있고 영어에 빠진 키"가 없는지 검사.

## 7. 품질 보증 (Claude가 직접 수행)

- **파이썬 단위 테스트(pytest):** 크기 공식, AST 추적, 진단 규칙(합성 곡선), 실험 기록, export 코드 생성.
- **실제 프레임워크 대조 테스트:** 레이어·인자 조합을 대량 생성해 정적 계산 결과와 실제 torch(CPU)/tf 실행 결과가 100% 같은지 비교. 다르면 실패.
- **확장 테스트:** 단위 테스트 + 실제 VS Code를 띄우는 통합 테스트(`@vscode/test-electron`)로 인레이 힌트·진단·빠른 수정·호버·CodeLens·명령·트리뷰가 예제 파일에서 기대대로 나오는지 확인.
- **웹뷰 확인:** 학습 닥터·모델 그림·비교·에러 카드 화면을 브라우저로 렌더링해 스크린샷으로 직접 눈으로 점검(밝은/어두운 테마, 좁은 폭).
- **예제 5개로 처음부터 끝까지:** MNIST MLP(torch), CIFAR CNN(torch), Keras CNN, 크기 에러 예제, 학습 문제 5종 재현 예제를 실제로 돌려 F1~F7 전 흐름 확인.
- **설치 테스트:** 패키징한 VSIX를 깨끗한 VS Code 프로필에 설치해 "설치 → 예제 열기 → 크기 표시 → 학습 → 진단 → 기록 → 비교" 확인. 다른 OS(Windows·Linux)는 GitHub Actions에서 같은 통합 테스트 실행(명세의 "3개 OS 설치 테스트" — 실제 화면 확인은 macOS에서만 가능하다는 한계를 문서에 명시).
- **CI:** GitHub Actions — OS 3종 × Python(3.9, 3.12) × torch(2.x 최저 지원·최신) / tf(2.x 최신) 조합.
- 단계마다 Git 커밋(명세 리스크 대응).

## 8. 배포·운영

- **GitHub 저장소** `mrpark10/DeepFriend`: 개발 중에는 **비공개**, 정식 출시 직전에 공개로 전환(사용자 결정). 공개 후 Issues가 문의·오류 신고 창구가 되므로, 공개 전환은 출시 체크리스트 항목으로 둔다.
- **게시자 ID:** `mrpark10` → 확장 ID `mrpark10.deepfriend` (사용자 결정). Open VSX 네임스페이스도 `mrpark10`.
- **문의 이메일:** sk92977852@gmail.com (사용자 결정) — 개인정보처리방침·마켓플레이스·README에 표기.
- **릴리스 자동화:** 태그 푸시 → CI 통과 → VS Code Marketplace(`vsce`), Open VSX(`ovsx`), PyPI(GitHub Trusted Publishing — 토큰 저장 불필요) 동시 게시.
- **게시자 계정 (사용자가 직접):** VS Code Marketplace 게시자 생성 + 토큰, Open VSX(Eclipse 계정·게시 동의서) 토큰, PyPI 계정 + Trusted Publisher 등록. Claude가 화면별 안내 문서를 만들고, 토큰은 사용자가 GitHub Secrets에 넣는다.
- **마켓플레이스 등록 정보:** 아이콘, 한·영 설명, 기능별 스크린샷/GIF, 키워드(pytorch, tensorflow, keras, deep learning, 한국어, 딥러닝, 교육 등), 카테고리(Machine Learning, Education, Linters, Visualization). 시연 영상은 GIF로 대체 가능한지 확인하고, 불가하면 사용자에게 녹화를 요청.
- **문서:** README(설치·기능별 사용법·지원 범위·FAQ), `docs/` 사용법, CHANGELOG.
- **법적 문서:** 개인정보처리방침·이용약관(한/영) — 수집 정보 없음, 모든 기록은 로컬, AI 사용 시 에러·코드 조각이 사용자가 고른 AI 제공자에게 전송됨을 명시. 저장소에 두고 README·마켓플레이스에서 링크.
- **라이선스:** MIT **[Claude 판단]** — 사용·기여 장벽이 낮아 확산에 유리.

## 9. 빌더에게 맡기는 것
- 진단 규칙의 정확한 임계값(예제로 조정), 차트 라이브러리 선택(동봉 가능하고 가벼운 것), 웹뷰 디자인 세부, 파일 구조 세부, 테스트 데이터 생성 방식, 문구 다듬기.

## 10. 열린 질문
- F6 PyTorch 변환기의 OS 지원 범위 — 구현 시작 시 재확인(명세의 열린 질문 그대로).
