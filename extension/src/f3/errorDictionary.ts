import { pick } from '../i18n/i18n';
import type { ParsedError } from './traceback';

interface L2 {
  ko: string;
  en: string;
}

const L = (ko: string, en: string): L2 => ({ ko, en });

export type ErrorAction = 'installModule' | 'goToLine' | 'search';

export interface Entry {
  id: string;
  /** Return the variables for the copy when this entry matches, else null. */
  match: (type: string, message: string) => Record<string, string> | null;
  title: L2;
  meaning: L2;
  cause: L2;
  fix: L2;
  actions?: ErrorAction[];
}

export interface Explanation {
  id: string;
  vars: Record<string, string>;
  actions: ErrorAction[];
  entry: Entry;
}

const PIP_NAMES: Record<string, string> = {
  cv2: 'opencv-python',
  sklearn: 'scikit-learn',
  PIL: 'pillow',
  yaml: 'pyyaml',
  skimage: 'scikit-image',
  bs4: 'beautifulsoup4',
  dotenv: 'python-dotenv',
  google: 'google-api-python-client',
  attr: 'attrs',
  Crypto: 'pycryptodome',
  serial: 'pyserial',
  tensorflow_datasets: 'tensorflow-datasets',
};

function re(pattern: RegExp, names: string[]): (type: string, message: string) => Record<string, string> | null {
  return (_type, message) => {
    const m = pattern.exec(message);
    if (!m) {
      return null;
    }
    const vars: Record<string, string> = {};
    names.forEach((n, i) => {
      if (n) {
        vars[n] = m[i + 1] ?? '';
      }
    });
    return vars;
  };
}

function typeIs(...types: string[]): (type: string, message: string) => Record<string, string> | null {
  return (type) => (types.includes(type.split('.').pop() ?? type) ? {} : null);
}

function anyOf(...fns: ((type: string, message: string) => Record<string, string> | null)[]) {
  return (type: string, message: string) => {
    for (const f of fns) {
      const v = f(type, message);
      if (v) {
        return v;
      }
    }
    return null;
  };
}

export const ENTRIES: Entry[] = [
  {
    id: 'shape_matmul',
    match: anyOf(
      re(/mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)/, ['rows', 'got', 'expected', 'out']),
      re(/a and b must have same reduction dim, but got \[[\d, ]*?(\d+)\] X \[(\d+), [\d, ]+\]/, ['got', 'expected']),
    ),
    title: L('크기 불일치: Linear 층에 들어가는 숫자 개수가 달라요', 'Shape mismatch: wrong number of inputs for a Linear layer'),
    meaning: L('앞 레이어가 {got}개의 숫자를 내보내는데, 다음 Linear 층은 {expected}개를 받도록 만들어져 있어요.', 'The previous layer outputs {got} values, but the next Linear layer was built to take {expected}.'),
    cause: L('보통 Flatten(또는 view) 다음 첫 Linear의 in_features를 잘못 계산했거나, 입력 이미지 크기가 생각과 달라서 생겨요.', 'Usually the first Linear after Flatten/view has a miscalculated in_features, or the input image size is different from what you assumed.'),
    fix: L('그 Linear 층의 in_features를 {got}로 바꿔 보세요. DeepFriend 크기 표시(# df: input 주석)를 쓰면 실행 전에 바로 알 수 있어요.', 'Change that Linear layer’s in_features to {got}. DeepFriend shape hints (# df: input comment) catch this before running.'),
  },
  {
    id: 'shape_conv_channels',
    match: re(/Given groups=\d+, weight of size \[[\d, ]+\], expected input\[[\d, ]+\] to have (\d+) channels, but got (\d+) channels/, ['expected', 'got']),
    title: L('크기 불일치: 합성곱 층의 채널 수가 달라요', 'Shape mismatch: wrong number of channels for a Conv layer'),
    meaning: L('Conv 층은 채널이 {expected}개인 입력을 기대하는데, 실제로는 {got}개짜리가 들어왔어요.', 'The Conv layer expects {expected} input channels but received {got}.'),
    cause: L('흑백(1채널) 이미지를 컬러(3채널)용 층에 넣었거나, 앞 Conv의 out_channels와 이 층의 in_channels가 다를 때 생겨요. 이미지 축 순서가 (N, H, W, C)인 경우에도 생겨요.', 'A grayscale (1-channel) image went into a layer built for color (3 channels), or the previous out_channels differs from this in_channels. Also happens with (N, H, W, C) ordering.'),
    fix: L('이 층의 in_channels를 {got}로 바꾸거나, 입력 데이터의 채널 수를 맞춰 주세요. 채널이 맨 뒤에 있다면 x.permute(0, 3, 1, 2)로 순서를 바꿔요.', 'Set in_channels to {got} or fix the input channels. If channels are last, use x.permute(0, 3, 1, 2).'),
  },
  {
    id: 'shape_tensor_size',
    match: re(/The size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\) at non-singleton dimension (\d+)/, ['a', 'b', 'dim']),
    title: L('크기 불일치: 두 텐서의 크기가 달라서 계산할 수 없어요', 'Shape mismatch: two tensors have different sizes'),
    meaning: L('{dim}번 축에서 한쪽은 크기가 {a}, 다른 쪽은 {b}라서 더하기·곱하기 같은 계산을 할 수 없어요.', 'At dimension {dim} one tensor has size {a} and the other {b}, so element-wise math is impossible.'),
    cause: L('손실 함수에 넣는 예측과 정답의 모양이 다르거나(예: (N, 1)과 (N,)), 출력 크기와 정답 개수가 다를 때 자주 생겨요.', 'Often the prediction and the target given to a loss have different shapes (e.g. (N, 1) vs (N,)), or the output size differs from the labels.'),
    fix: L('계산하는 두 텐서의 .shape를 print로 찍어 보고, 같은 모양이 되게 view·squeeze·unsqueeze로 맞춰 주세요.', 'Print both tensors’ .shape and make them match with view/squeeze/unsqueeze.'),
  },
  {
    id: 'shape_view',
    match: re(/shape '(\[[^\]]*\])' is invalid for input of size (\d+)/, ['shape', 'size']),
    title: L('크기 불일치: view/reshape 모양이 맞지 않아요', 'Shape mismatch: view/reshape target is impossible'),
    meaning: L('원소가 {size}개인 데이터를 {shape} 모양으로 바꿀 수 없어요. 바꾸기 전후의 원소 수가 같아야 해요.', 'Data with {size} elements cannot become {shape}; the element count must stay the same.'),
    cause: L('view에 직접 숫자를 적었는데 입력 크기나 배치 크기가 달라졌을 때 생겨요. 마지막 배치가 작을 때도 자주 생겨요.', 'Happens when numbers are hard-coded in view but the input or batch size changed — often on the smaller last batch.'),
    fix: L('배치 크기는 고정하지 말고 x.view(x.size(0), -1)처럼 -1을 쓰세요.', 'Do not hard-code the batch size; use x.view(x.size(0), -1).'),
  },
  {
    id: 'shape_target_batch',
    match: re(/Expected input batch_size \((\d+)\) to match target batch_size \((\d+)\)/, ['input', 'target']),
    title: L('크기 불일치: 예측 개수와 정답 개수가 달라요', 'Shape mismatch: predictions and labels have different batch sizes'),
    meaning: L('모델 출력은 {input}개인데 정답(라벨)은 {target}개예요.', 'The model produced {input} predictions but there are {target} labels.'),
    cause: L('forward 안에서 view에 숫자를 잘못 넣어 배치 크기가 바뀌었거나, 데이터와 라벨을 다른 배치에서 꺼냈을 때 생겨요.', 'A wrong view inside forward changed the batch size, or inputs and labels come from different batches.'),
    fix: L('forward의 view/reshape를 x.view(x.size(0), -1)로 바꾸고, 같은 배치에서 (입력, 라벨)을 함께 꺼냈는지 확인하세요.', 'Use x.view(x.size(0), -1) in forward and make sure inputs and labels come from the same batch.'),
  },
  {
    id: 'shape_keras_input',
    match: anyOf(
      re(/Expected shape (\([^)]*\)), but input has incompatible shape (\([^)]*\))/, ['expected', 'got']),
      re(/expected axis -?\d+ of input shape to have value (\d+), but received input with shape (\([^)]*\))/, ['expected', 'got']),
      re(/expected min_ndim=(\d+), found ndim=(\d+)/, ['expected', 'got']),
    ),
    title: L('크기 불일치: 모델 입력 모양이 달라요 (Keras)', 'Shape mismatch: wrong model input shape (Keras)'),
    meaning: L('모델은 {expected} 모양의 입력을 기대하는데 {got} 모양이 들어왔어요.', 'The model expects input shaped {expected} but got {got}.'),
    cause: L('이미지를 펼치지 않고 Dense 모델에 넣었거나, 흑백 이미지에 채널 축(…, 1)이 빠졌을 때 자주 생겨요.', 'Often images were not flattened for a Dense model, or the channel axis (…, 1) is missing for grayscale images.'),
    fix: L('데이터를 reshape로 모델 입력 모양에 맞추거나(예: x.reshape(-1, 784)), 모델 맨 앞에 layers.Flatten()을 넣어 주세요.', 'Reshape the data to the model input (e.g. x.reshape(-1, 784)) or put layers.Flatten() first.'),
  },
  {
    id: 'shape_keras_target',
    match: anyOf(
      re(/target\.shape=(\([^)]*\)), output\.shape=(\([^)]*\))/, ['target', 'output']),
      re(/Shapes (\([^)]*\)) and (\([^)]*\)) are incompatible/, ['target', 'output']),
    ),
    title: L('크기 불일치: 정답(라벨) 모양과 모델 출력 모양이 달라요 (Keras)', 'Shape mismatch: labels and model output differ (Keras)'),
    meaning: L('정답은 {target} 모양인데 모델 출력은 {output} 모양이에요.', 'Labels are shaped {target} but the model outputs {output}.'),
    cause: L('라벨이 클래스 번호(0, 1, 2…)인데 categorical_crossentropy를 썼을 때 가장 흔해요.', 'Most common: labels are class numbers (0, 1, 2…) but categorical_crossentropy was used.'),
    fix: L("손실을 loss='sparse_categorical_crossentropy'로 바꾸거나, 라벨을 keras.utils.to_categorical로 원-핫으로 바꿔 주세요.", "Use loss='sparse_categorical_crossentropy', or one-hot the labels with keras.utils.to_categorical."),
  },
  {
    id: 'cuda_oom',
    match: anyOf(re(/CUDA out of memory\. Tried to allocate ([\d.]+ [KMGT]i?B)/, ['size']), (type) => (/OutOfMemoryError$/.test(type) ? { size: '?' } : null)),
    title: L('GPU 메모리가 부족해요', 'Out of GPU memory'),
    meaning: L('GPU에 {size}를 더 올리려고 했는데 남은 메모리가 부족해요.', 'Tried to put {size} more on the GPU but there is not enough memory.'),
    cause: L('배치 크기나 이미지 크기가 너무 크거나, 모델이 너무 커요. loss를 리스트에 그대로 쌓아 두면(loss.item() 없이) 메모리가 계속 늘어나기도 해요.', 'The batch or images are too large, or the model is too big. Storing losses without loss.item() also keeps growing memory.'),
    fix: L('batch_size를 절반으로 줄여 보세요. 평가할 때는 with torch.no_grad():를 쓰고, 기록할 때는 loss.item()으로 숫자만 저장하세요.', 'Halve batch_size. Use with torch.no_grad(): for evaluation and store loss.item() instead of the tensor.'),
  },
  {
    id: 'tf_oom',
    match: (type, message) => (/ResourceExhaustedError$/.test(type) || /OOM when allocating tensor/.test(message) ? {} : null),
    title: L('GPU 메모리가 부족해요 (TensorFlow)', 'Out of GPU memory (TensorFlow)'),
    meaning: L('텐서를 만들 메모리가 부족해요.', 'There is not enough memory to allocate a tensor.'),
    cause: L('batch_size나 입력 이미지가 너무 크거나 모델이 너무 커요.', 'batch_size, image size or model is too large.'),
    fix: L('model.fit의 batch_size를 줄이거나 이미지 크기를 줄여 보세요.', 'Lower batch_size in model.fit or use smaller images.'),
  },
  {
    id: 'dtype_matmul',
    match: anyOf(re(/must have the same dtype, but got (\w+) and (\w+)/, ['a', 'b']), re(/expected scalar type (\w+) but found (\w+)/, ['b', 'a'])),
    title: L('자료형 불일치: 숫자 종류(float32/float64)가 달라요', 'Dtype mismatch: float32 vs float64'),
    meaning: L('한쪽은 {a}, 다른 쪽은 {b} 자료형이라 함께 계산할 수 없어요.', 'One side is {a} and the other {b}, so they cannot be combined.'),
    cause: L('numpy 배열(float64)을 torch.from_numpy로 바꾸면 float64가 되는데, 모델 가중치는 float32라서 자주 생겨요.', 'numpy arrays are float64; torch.from_numpy keeps that, while model weights are float32.'),
    fix: L('입력을 x.float()로 바꾸거나, numpy에서 x.astype("float32")로 바꾼 뒤 텐서로 만드세요.', 'Convert inputs with x.float(), or use x.astype("float32") in numpy first.'),
  },
  {
    id: 'dtype_target',
    match: anyOf(re(/expected target dtype to be Long or Byte, but got (\w+)/, ['got']), re(/expected scalar type Long but found (\w+)/, ['got'])),
    title: L('자료형 불일치: 정답(라벨)은 정수(long)여야 해요', 'Dtype mismatch: labels must be integers (long)'),
    meaning: L('분류 손실(CrossEntropy/NLL)은 정답을 정수(long) 클래스 번호로 받는데, 지금은 {got} 자료형이에요.', 'Classification losses take integer (long) class ids, but the labels are {got}.'),
    cause: L('라벨을 float로 만들었거나 numpy int32 배열을 그대로 텐서로 바꿨을 때 생겨요.', 'Labels were created as float or converted from int32 numpy arrays.'),
    fix: L('정답을 labels.long()으로 바꿔서 손실 함수에 넣어 주세요.', 'Pass labels.long() to the loss.'),
  },
  {
    id: 'device_mismatch',
    match: re(/Expected all tensors to be on the same device/, []),
    title: L('장치 불일치: 텐서들이 서로 다른 장치(CPU/GPU)에 있어요', 'Device mismatch: tensors are on different devices'),
    meaning: L('함께 계산하는 텐서 중 일부는 GPU에, 일부는 CPU에 있어요.', 'Some tensors are on the GPU and others on the CPU.'),
    cause: L('모델은 .to(device)로 옮겼지만 입력이나 정답을 옮기지 않았을 때 가장 흔해요.', 'Most often the model was moved with .to(device) but inputs or labels were not.'),
    fix: L('학습 루프에서 x, y = x.to(device), y.to(device)처럼 입력과 정답도 같은 장치로 옮겨 주세요.', 'In the loop, move data too: x, y = x.to(device), y.to(device).'),
  },
  {
    id: 'device_input_weight',
    match: anyOf(re(/Input type \((.+?)\) and weight type \((.+?)\) should be the same/, ['input', 'weight']), re(/Tensor for argument \w+ is on (\w+) but expected on (\w+)/, ['input', 'weight'])),
    title: L('장치 불일치: 입력과 모델 가중치가 다른 장치에 있어요', 'Device mismatch: input and model weights are on different devices'),
    meaning: L('입력 데이터와 모델 가중치가 서로 다른 장치(CPU/GPU)에 있어요.', 'The input and the model weights live on different devices.'),
    cause: L('model.to(device)를 빼먹었거나, 입력만 GPU로 옮겼을 때 생겨요.', 'model.to(device) was missed, or only the input was moved.'),
    fix: L('model = model.to(device)와 x = x.to(device)를 둘 다 해 주세요.', 'Do both model = model.to(device) and x = x.to(device).'),
  },
  {
    id: 'cuda_unavailable',
    match: re(/(Torch not compiled with CUDA enabled|Found no NVIDIA driver|no CUDA GPUs are available|CUDA driver initialization failed)/, ['reason']),
    title: L('이 컴퓨터에서는 CUDA(GPU)를 쓸 수 없어요', 'CUDA (GPU) is not available on this computer'),
    meaning: L('코드에서 .cuda()나 device="cuda"를 썼지만 이 컴퓨터의 PyTorch는 NVIDIA GPU를 쓸 수 없어요.', 'The code uses .cuda() or device="cuda", but this PyTorch cannot use an NVIDIA GPU.'),
    cause: L('맥이나 GPU가 없는 컴퓨터이거나, CPU 전용 PyTorch가 설치돼 있어요.', 'This is a Mac or a machine without an NVIDIA GPU, or a CPU-only PyTorch is installed.'),
    fix: L('device = "cuda" if torch.cuda.is_available() else "cpu"로 장치를 고르고 .to(device)를 쓰세요. 맥이라면 "mps"도 쓸 수 있어요.', 'Pick the device with device = "cuda" if torch.cuda.is_available() else "cpu" and use .to(device). On Macs, "mps" also works.'),
  },
  {
    id: 'module_not_found',
    match: (type, message) => {
      const m = /No module named '([^']+)'/.exec(message);
      if (!m || !/(ModuleNotFoundError|ImportError)$/.test(type)) {
        return null;
      }
      const top = m[1].split('.')[0];
      return { module: m[1], pip: PIP_NAMES[top] ?? top };
    },
    title: L('모듈 없음: {module} 패키지가 설치돼 있지 않아요', 'Module not found: {module} is not installed'),
    meaning: L('파이썬이 {module}을(를) 찾지 못했어요.', 'Python cannot find {module}.'),
    cause: L('패키지를 설치하지 않았거나, VS Code가 다른 파이썬(가상환경)을 쓰고 있을 때 생겨요.', 'The package is not installed, or VS Code is using a different Python (virtual environment).'),
    fix: L('아래 "설치" 버튼을 누르면 지금 쓰는 파이썬에 pip install {pip}를 실행해요. 설치했는데도 같다면 VS Code 오른쪽 아래에서 파이썬 인터프리터를 확인해 보세요.', 'Press “Install” to run pip install {pip} for the current Python. If it persists, check the selected interpreter in VS Code.'),
    actions: ['installModule'],
  },
  {
    id: 'indentation',
    match: typeIs('IndentationError', 'TabError'),
    title: L('들여쓰기 오류', 'Indentation error'),
    meaning: L('파이썬은 들여쓰기(줄 앞 공백)로 코드 묶음을 구분하는데, 그 칸이 맞지 않아요.', 'Python uses indentation to group code, and it does not line up here.'),
    cause: L('def·for·if·class 다음 줄을 들여쓰지 않았거나, 탭과 스페이스를 섞어 썼을 때 생겨요.', 'A line after def/for/if/class is not indented, or tabs and spaces are mixed.'),
    fix: L('에러 난 줄과 그 위 줄의 들여쓰기를 스페이스 4칸으로 맞춰 주세요.', 'Indent the reported line and the line above with 4 spaces.'),
  },
  {
    id: 'syntax_error',
    match: typeIs('SyntaxError'),
    title: L('문법 오류', 'Syntax error'),
    meaning: L('파이썬이 코드를 읽다가 문법에 맞지 않는 부분을 만났어요.', 'Python found code that does not follow its grammar.'),
    cause: L('괄호·따옴표를 닫지 않았거나, 콜론(:)을 빼먹었거나, 오타가 있을 때 생겨요. 진짜 원인은 표시된 줄의 바로 윗줄에 있는 경우가 많아요.', 'Unclosed brackets/quotes, a missing colon or a typo. The real cause is often the line just above.'),
    fix: L('표시된 줄과 바로 위 줄의 괄호·따옴표·콜론을 확인해 보세요.', 'Check brackets, quotes and colons on that line and the one above.'),
  },
  {
    id: 'name_error',
    match: re(/name '(\w+)' is not defined/, ['name']),
    title: L('이름 오류: {name}을(를) 찾을 수 없어요', 'Name error: {name} is not defined'),
    meaning: L('{name}이라는 변수나 함수가 아직 만들어지지 않았어요.', 'There is no variable or function called {name} yet.'),
    cause: L('오타가 있거나, 정의하기 전에 썼거나, import를 빼먹었을 때 생겨요. 노트북이라면 그 셀을 아직 실행하지 않았을 수도 있어요.', 'A typo, used before being defined, or a missing import. In notebooks the defining cell may not have run.'),
    fix: L('철자를 확인하고, 필요한 import나 정의가 이 줄보다 먼저 실행되는지 확인하세요.', 'Check the spelling and make sure the import/definition runs before this line.'),
  },
  {
    id: 'attribute_none',
    match: re(/'NoneType' object has no attribute '(\w+)'/, ['attr']),
    title: L('값이 None이에요', 'The value is None'),
    meaning: L('None(아무것도 없음)에서 .{attr}을(를) 꺼내려고 했어요.', 'Tried to access .{attr} on None.'),
    cause: L('함수가 return을 하지 않았거나(예: forward에서 return 빼먹음), 파일을 못 읽어서 None이 돌아왔을 때 생겨요.', 'A function returned nothing (e.g. forward without return) or a load failed and returned None.'),
    fix: L('바로 앞에서 그 값을 만든 함수에 return이 있는지, 파일 경로가 맞는지 확인하세요.', 'Check that the producing function returns a value and that file paths are correct.'),
  },
  {
    id: 'attribute_error',
    match: re(/'(\w+)' object has no attribute '(\w+)'/, ['obj', 'attr']),
    title: L('속성 오류: {obj}에는 {attr}이(가) 없어요', 'Attribute error: {obj} has no {attr}'),
    meaning: L('{obj} 객체에 {attr}이라는 속성이나 메서드가 없어요.', '{obj} objects have no attribute or method named {attr}.'),
    cause: L('오타이거나, __init__에서 self.{attr}을(를) 만들지 않았거나, 다른 종류의 객체를 쓰고 있을 때 생겨요.', 'A typo, self.{attr} was never created in __init__, or this is a different kind of object.'),
    fix: L('철자를 확인하고 __init__에 self.{attr} = ... 가 있는지 확인하세요.', 'Check the spelling and that __init__ sets self.{attr} = ...'),
  },
  {
    id: 'type_args',
    match: (type, message) => (/TypeError$/.test(type) && /(takes \d+ positional arguments? but \d+ (?:was|were) given|missing \d+ required positional argument)/.test(message) ? {} : null),
    title: L('인자 개수가 맞지 않아요', 'Wrong number of arguments'),
    meaning: L('함수(또는 forward)에 넘긴 값의 개수가 정의된 개수와 달라요.', 'The number of values passed differs from the function definition.'),
    cause: L('forward(self, x)인데 model(x, y)처럼 두 개를 넘겼거나, self를 빼먹었을 때 생겨요.', 'forward(self, x) was called with two values like model(x, y), or self is missing.'),
    fix: L('함수 정의와 호출하는 곳의 인자 개수를 맞춰 주세요. 메서드 정의에는 첫 번째로 self가 있어야 해요.', 'Match the arguments in the definition and the call. Methods need self first.'),
  },
  {
    id: 'index_error',
    match: typeIs('IndexError'),
    title: L('인덱스 범위를 벗어났어요', 'Index out of range'),
    meaning: L('목록이나 텐서에 없는 번호의 칸을 꺼내려고 했어요.', 'Tried to access a position that does not exist.'),
    cause: L('번호는 0부터 시작하니까 길이가 n이면 마지막 번호는 n−1이에요. 클래스 번호가 출력 개수보다 클 때도 생겨요.', 'Indexes start at 0, so the last one is n−1. Also happens when a label is larger than the number of outputs.'),
    fix: L('len()이나 .shape로 크기를 찍어 보고 번호를 확인하세요.', 'Print len() or .shape and check the index.'),
  },
  {
    id: 'key_error',
    match: (type, message) => (/KeyError$/.test(type) ? { key: message || '?' } : null),
    title: L('키 오류: {key} 키가 없어요', 'Key error: {key} not found'),
    meaning: L('딕셔너리(또는 state_dict)에 {key}라는 키가 없어요.', 'The dictionary (or state_dict) has no key {key}.'),
    cause: L('오타이거나, 저장할 때와 불러올 때 이름이 다를 때 생겨요.', 'A typo, or the names differ between saving and loading.'),
    fix: L('print(d.keys())로 실제 키 목록을 확인하세요. 모르면 d.get(키, 기본값)을 써도 돼요.', 'Print d.keys() to see the real keys, or use d.get(key, default).'),
  },
  {
    id: 'file_not_found',
    match: re(/No such file or directory: '([^']+)'/, ['path']),
    title: L('파일을 찾을 수 없어요: {path}', 'File not found: {path}'),
    meaning: L('{path} 위치에 파일이나 폴더가 없어요.', 'There is no file or folder at {path}.'),
    cause: L('상대 경로는 "지금 터미널이 있는 폴더" 기준이라, 다른 폴더에서 실행하면 못 찾아요.', 'Relative paths are based on the terminal’s current folder, so running from elsewhere fails.'),
    fix: L('파일이 실제로 있는지 확인하고, 스크립트 위치 기준 경로를 쓰세요: os.path.join(os.path.dirname(__file__), "data")', 'Check the file exists and use a path relative to the script: os.path.join(os.path.dirname(__file__), "data").'),
  },
  {
    id: 'zero_division',
    match: typeIs('ZeroDivisionError'),
    title: L('0으로 나눴어요', 'Division by zero'),
    meaning: L('어떤 값을 0으로 나누려고 했어요.', 'Something was divided by zero.'),
    cause: L('데이터 개수나 배치 수가 0일 때(데이터를 못 읽었을 때) 정확도를 계산하면서 자주 생겨요.', 'Often computing accuracy when the dataset or batch count is 0 (data failed to load).'),
    fix: L('나누는 값이 0이 아닌지 확인하고, 데이터가 제대로 읽혔는지 len(dataset)을 찍어 보세요.', 'Check the divisor and print len(dataset) to confirm data loaded.'),
  },
];

/** Match a parsed error against the offline dictionary. */
export function explain(err: ParsedError): Explanation | null {
  for (const entry of ENTRIES) {
    const vars = entry.match(err.type, err.message);
    if (vars) {
      return { id: entry.id, vars, actions: ['goToLine', ...(entry.actions ?? [])], entry };
    }
  }
  return null;
}

function fill(text: string, vars: Record<string, string>): string {
  return text.replace(/\{([a-z]+)\}/g, (m, k: string) => vars[k] ?? '?');
}

export interface RenderedExplanation {
  title: string;
  meaning: string;
  cause: string;
  fix: string;
}

export function renderExplanation(x: Explanation): RenderedExplanation {
  const e = x.entry;
  return {
    title: fill(pick(e.title), x.vars),
    meaning: fill(pick(e.meaning), x.vars),
    cause: fill(pick(e.cause), x.vars),
    fix: fill(pick(e.fix), x.vars),
  };
}
