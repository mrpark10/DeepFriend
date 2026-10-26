import { pick, StringKey, t } from '../i18n/i18n';
import type { ArgLocation } from '../server/types';
import type { DoctorAction, Finding } from './doctorRules';

export interface RenderedFinding {
  what: string;
  why: string;
  try: string;
}

export function renderFinding(f: Finding): RenderedFinding {
  const args = f.args as Record<string, number>;
  return {
    what: t(`doctor.${f.id}.what` as StringKey, args),
    why: t(`doctor.${f.id}.why` as StringKey, args),
    try: t(`doctor.${f.id}.try` as StringKey, args),
  };
}

export function actionLabel(a: DoctorAction): string {
  switch (a.kind) {
    case 'setLr':
      return t(a.factor < 1 ? 'doctor.action.lrDown' : 'doctor.action.lrUp');
    case 'setBatch':
      return t('doctor.action.batchUp');
    case 'snippet':
      return t(`doctor.action.${a.id}` as StringKey);
    case 'showSamples':
      return t('doctor.action.showSamples');
  }
}

/** Write a float the way it usually appears in code: 0.003, 1e-4, 1.0. */
export function formatNumber(v: number): string {
  const r = Number(v.toPrecision(3));
  if (r >= 1e-3 || r === 0) {
    const s = String(r);
    return Number.isInteger(r) ? s + '.0' : s;
  }
  return r.toExponential().replace('e-0', 'e-');
}

export interface TextEditSpec {
  line: number;
  col: number;
  endCol: number;
  text: string;
}

/** The text edit that applies a doctor action at a located argument, or null if impossible. */
export function editForAction(loc: ArgLocation, a: DoctorAction): TextEditSpec | null {
  if (a.kind !== 'setLr' && a.kind !== 'setBatch') {
    return null;
  }
  const base = loc.insert ? loc.defaultValue : loc.value;
  if (base === null || base === undefined) {
    return null;
  }
  const next = a.kind === 'setBatch' ? String(Math.max(1, Math.round(base * a.factor))) : formatNumber(base * a.factor);
  if (loc.insert) {
    return { line: loc.line, col: loc.col, endCol: loc.endCol, text: `${loc.hasArgs ? ', ' : ''}${loc.name}=${next}` };
  }
  return { line: loc.line, col: loc.col, endCol: loc.endCol, text: next };
}

type Snip = { ko: string; en: string };
const SNIPPETS: Record<string, { pytorch: Snip; tensorflow: Snip }> = {
  earlyStopping: {
    pytorch: {
      ko: `# 조기 종료(early stopping): 검증 loss가 patience번 연속 좋아지지 않으면 멈춰요.
best_val = float("inf")
patience, bad_epochs = 3, 0
for epoch in range(num_epochs):
    train_one_epoch(model)
    val_loss = evaluate(model)
    if val_loss < best_val:
        best_val, bad_epochs = val_loss, 0
        torch.save(model.state_dict(), "best_model.pt")  # 가장 좋았던 상태 저장
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("검증 loss가 더 좋아지지 않아서 멈춰요")
            break
model.load_state_dict(torch.load("best_model.pt"))
`,
      en: `# Early stopping: stop when validation loss has not improved for \`patience\` epochs.
best_val = float("inf")
patience, bad_epochs = 3, 0
for epoch in range(num_epochs):
    train_one_epoch(model)
    val_loss = evaluate(model)
    if val_loss < best_val:
        best_val, bad_epochs = val_loss, 0
        torch.save(model.state_dict(), "best_model.pt")  # keep the best weights
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("Validation loss stopped improving")
            break
model.load_state_dict(torch.load("best_model.pt"))
`,
    },
    tensorflow: {
      ko: `# 조기 종료(early stopping): 검증 loss가 patience번 연속 좋아지지 않으면 멈춰요.
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
model.fit(x_train, y_train, validation_split=0.2, epochs=50, callbacks=[early_stop])
`,
      en: `# Early stopping: stop when validation loss has not improved for \`patience\` epochs.
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
model.fit(x_train, y_train, validation_split=0.2, epochs=50, callbacks=[early_stop])
`,
    },
  },
  dropout: {
    pytorch: {
      ko: `# Dropout: 학습할 때 일부 뉴런을 무작위로 꺼서 외우기(과적합)를 막아요.
self.classifier = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),   # ← 추가
    nn.Linear(256, num_classes),
)
# 평가할 때는 model.eval()을 꼭 불러야 Dropout이 꺼져요.
`,
      en: `# Dropout: randomly turns off neurons during training to prevent memorizing (overfitting).
self.classifier = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),   # <- added
    nn.Linear(256, num_classes),
)
# Call model.eval() before evaluating so Dropout is turned off.
`,
    },
    tensorflow: {
      ko: `# Dropout: 학습할 때 일부 뉴런을 무작위로 꺼서 외우기(과적합)를 막아요.
model = keras.Sequential([
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),   # ← 추가
    layers.Dense(num_classes, activation="softmax"),
])
`,
      en: `# Dropout: randomly turns off neurons during training to prevent memorizing (overfitting).
model = keras.Sequential([
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),   # <- added
    layers.Dense(num_classes, activation="softmax"),
])
`,
    },
  },
  normalize: {
    pytorch: {
      ko: `# 입력 정규화: 픽셀 값을 0~1로 바꾸고 평균 0, 표준편차 1로 맞춰요.
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),                       # 0~255 → 0~1
    transforms.Normalize((0.5,), (0.5,)),        # 흑백. 컬러면 (0.5, 0.5, 0.5)
])
# 표 데이터라면: x = (x - x.mean(0)) / (x.std(0) + 1e-8)
`,
      en: `# Input normalization: scale pixels to 0-1, then to mean 0 / std 1.
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),                       # 0-255 -> 0-1
    transforms.Normalize((0.5,), (0.5,)),        # grayscale; use (0.5, 0.5, 0.5) for color
])
# For tabular data: x = (x - x.mean(0)) / (x.std(0) + 1e-8)
`,
    },
    tensorflow: {
      ko: `# 입력 정규화: 픽셀 값을 0~1로 바꿔요.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
# 또는 모델 맨 앞에 layers.Rescaling(1.0 / 255)
`,
      en: `# Input normalization: scale pixels to 0-1.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
# or put layers.Rescaling(1.0 / 255) first in the model
`,
    },
  },
  eps: {
    pytorch: {
      ko: `# log나 나눗셈에 아주 작은 값(eps)을 더해 log(0)·0으로 나누기를 막아요.
eps = 1e-8
loss = -(target * torch.log(pred + eps)).sum(dim=1).mean()
ratio = a / (b + eps)
# 가능하면 직접 계산하지 말고 nn.CrossEntropyLoss, nn.BCEWithLogitsLoss를 쓰세요(안정적이에요).
`,
      en: `# Add a tiny value (eps) inside log and divisions to avoid log(0) and division by zero.
eps = 1e-8
loss = -(target * torch.log(pred + eps)).sum(dim=1).mean()
ratio = a / (b + eps)
# Prefer nn.CrossEntropyLoss / nn.BCEWithLogitsLoss over hand-written losses (more stable).
`,
    },
    tensorflow: {
      ko: `# log나 나눗셈에 아주 작은 값(eps)을 더해 log(0)·0으로 나누기를 막아요.
eps = 1e-8
loss = -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_pred + eps), axis=1))
# 가능하면 직접 계산하지 말고 keras 내장 손실 함수를 쓰세요.
`,
      en: `# Add a tiny value (eps) inside log and divisions to avoid log(0) and division by zero.
eps = 1e-8
loss = -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_pred + eps), axis=1))
# Prefer the built-in Keras losses over hand-written ones.
`,
    },
  },
};

export function snippet(id: string, framework: 'pytorch' | 'tensorflow' | null | undefined): string {
  const s = SNIPPETS[id];
  if (!s) {
    return '';
  }
  return pick(framework === 'tensorflow' ? s.tensorflow : s.pytorch);
}
