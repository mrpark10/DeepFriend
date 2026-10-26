// Offline concept dictionary for F7 hovers. Plain Korean (해요체) first, English second.

export interface L2 {
  ko: string;
  en: string;
}

export interface Concept {
  title: L2;
  what: { beginner: L2; normal: L2 };
  diagram?: L2;
  params: Record<string, L2>;
  doc: string;
}

const L = (ko: string, en: string): L2 => ({ ko, en });

const PT = (name: string): string => `https://pytorch.org/docs/stable/generated/torch.nn.${name}.html`;
const KR = (path: string): string => `https://keras.io/api/layers/${path}/`;

// ---- shared parameter texts ----------------------------------------------------------

const P = {
  in_channels: L('들어오는 데이터의 채널 수예요. 흑백 이미지는 1, 컬러(RGB)는 3이에요.', 'Number of channels coming in. 1 for grayscale, 3 for RGB images.'),
  out_channels: L('이 층이 만들어 내는 채널(필터) 수예요. 많을수록 더 많은 특징을 찾지만 계산이 늘어요.', 'Number of channels (filters) this layer produces.'),
  kernel_size: L('한 번에 훑어보는 창의 크기예요. 3이면 3×3 칸을 봐요.', 'Size of the sliding window. 3 means a 3×3 window.'),
  stride: L('창을 몇 칸씩 건너뛰며 움직일지예요. 2면 출력 크기가 대략 절반이 돼요.', 'How many steps the window moves each time. 2 roughly halves the output size.'),
  padding: L("가장자리에 0을 몇 칸 덧댈지예요. 'same'이면 크기가 그대로 유지돼요.", "How many zeros to pad around the edges. 'same' keeps the size."),
  dilation: L('창 안의 칸 사이를 벌려서 더 넓게 보는 정도예요. 보통 1이에요.', 'Spacing between kernel elements. Usually 1.'),
  groups: L('채널을 몇 묶음으로 나눠 따로 계산할지예요. 보통 1이에요.', 'Splits channels into groups computed separately. Usually 1.'),
  bias: L('출력에 더해지는 값(편향)을 학습할지예요. 바로 뒤에 BatchNorm이 오면 False로 두기도 해요.', 'Whether to learn an additive bias. Often False when BatchNorm follows.'),
  pool_kernel: L('한 번에 묶어서 줄일 창의 크기예요. 2면 2×2 칸을 하나로 줄여요.', 'Window size to pool. 2 turns each 2×2 block into one value.'),
  pool_stride: L('창을 몇 칸씩 옮길지예요. 안 적으면 kernel_size와 같아요.', 'Step of the window. Defaults to kernel_size.'),
  ceil_mode: L('나누어떨어지지 않을 때 크기를 올림할지(True) 버림할지(False)예요.', 'Round the output size up (True) or down (False).'),
  inplace: L('메모리를 아끼려고 입력을 그 자리에서 바꿀지예요. 결과는 같아요.', 'Modify the input in place to save memory. Same result.'),
  dim: L('어느 축(차원)을 따라 계산할지예요. 분류 문제의 출력이면 보통 dim=1이에요.', 'Which dimension to compute along. Usually dim=1 for classifier outputs.'),
  num_features: L('들어오는 채널 수예요. 바로 앞 층의 out_channels(또는 출력 크기)와 같아야 해요.', 'Number of channels coming in. Must match the previous layer output.'),
  eps: L('0으로 나누는 것을 막으려고 더하는 아주 작은 값이에요.', 'Tiny value added to avoid division by zero.'),
  momentum: L('평균·분산을 얼마나 빠르게 갱신할지예요. 보통 그대로 둬요.', 'How fast running statistics are updated. Usually left as is.'),
  hidden_size: L('기억(은닉 상태)의 크기예요. 클수록 더 많이 기억하지만 느려져요.', 'Size of the hidden state.'),
  input_size: L('한 시점에 들어오는 특징 수예요. 데이터의 마지막 크기와 같아야 해요.', 'Number of features per time step. Must match the last input dimension.'),
  num_layers: L('RNN 층을 몇 겹 쌓을지예요.', 'Number of stacked recurrent layers.'),
  batch_first: L('True면 입력 모양이 (배치, 시간, 특징)이에요. 헷갈리면 True를 추천해요.', 'If True, input is (batch, time, features). Recommended.'),
  bidirectional: L('앞에서 뒤로, 뒤에서 앞으로 양쪽으로 읽을지예요. True면 출력 크기가 두 배가 돼요.', 'Read the sequence in both directions. Doubles the output size.'),
  dropout_p: L('학습할 때 무작위로 끌 비율이에요. 0.5면 절반을 꺼요.', 'Fraction of values randomly zeroed during training.'),
  lr: L('학습률이에요. 한 번에 얼마나 크게 고칠지 정해요. 너무 크면 loss가 튀거나 NaN이 되고, 너무 작으면 거의 안 배워요.', 'Learning rate: how big each update is. Too large → unstable/NaN, too small → barely learns.'),
  weight_decay: L('가중치가 너무 커지지 않게 막는 정도예요. 과적합을 줄여 줘요.', 'Penalty that keeps weights small. Reduces overfitting.'),
  reduction: L("배치의 loss를 어떻게 합칠지예요. 'mean'(평균)이 기본이에요.", "How to combine losses over the batch. 'mean' by default."),
  units: L('출력 뉴런 수예요. 분류의 마지막 층이면 클래스 수와 같아야 해요.', 'Number of output neurons. Equals the number of classes for the last classifier layer.'),
  activation: L("층 뒤에 붙일 활성화 함수예요. 예: 'relu', 'softmax'.", "Activation applied after the layer, e.g. 'relu', 'softmax'."),
  filters: L('만들어 낼 채널(필터) 수예요.', 'Number of output channels (filters).'),
  strides: L('창을 몇 칸씩 건너뛸지예요. 2면 크기가 대략 절반이 돼요.', 'Step size of the window. 2 roughly halves the size.'),
  k_padding: L("'valid'는 덧대지 않아 크기가 줄고, 'same'은 크기를 유지해요.", "'valid' shrinks the size, 'same' keeps it."),
  pool_size: L('한 번에 묶어 줄일 창의 크기예요. 기본 2예요.', 'Pooling window size. Default 2.'),
  input_shape: L('첫 층에만 적는 입력 모양이에요. 배치 크기는 빼고 적어요. 예: (28, 28, 1)', 'Input shape for the first layer, without the batch size, e.g. (28, 28, 1).'),
  rate: L('학습할 때 무작위로 끌 비율이에요.', 'Fraction of inputs randomly dropped during training.'),
  num_embeddings: L('단어(토큰) 종류의 개수예요. 가장 큰 번호 + 1 이상이어야 해요.', 'Vocabulary size. Must be larger than the largest index.'),
  embedding_dim: L('단어 하나를 몇 개의 숫자로 표현할지예요.', 'Length of the vector for each token.'),
  in_features: L('들어오는 숫자의 개수예요. 바로 앞 층 출력의 마지막 크기와 같아야 해요.', 'Number of input features. Must equal the last size of the previous output.'),
  out_features: L('내보낼 숫자의 개수예요. 분류의 마지막 층이면 클래스 수예요.', 'Number of output features. Number of classes for the last layer.'),
  output_size: L('원하는 출력 크기예요. 입력 크기와 상관없이 항상 이 크기로 맞춰 줘요.', 'Target output size regardless of input size.'),
  start_dim: L('어느 축부터 펼칠지예요. 기본 1이라 배치 축(0)은 남겨 둬요.', 'First dimension to flatten. Default 1 keeps the batch dimension.'),
  end_dim: L('어느 축까지 펼칠지예요. 기본 -1(마지막)이에요.', 'Last dimension to flatten. Default -1 (last).'),
  normalized_shape: L('정규화할 마지막 축들의 크기예요. 보통 특징 수 하나를 적어요.', 'Size of the last dimension(s) to normalize.'),
  negative_slope: L('음수 쪽에서의 기울기예요. 0.01이면 음수를 100분의 1로 줄여요.', 'Slope for negative inputs.'),
  batch_size: L('한 번에 묶어서 학습할 데이터 개수예요. 메모리가 부족하면 줄이세요.', 'How many samples per batch. Lower it if memory runs out.'),
  shuffle: L('에폭마다 데이터 순서를 섞을지예요. 학습용은 True, 검증용은 False가 보통이에요.', 'Shuffle every epoch. True for training, False for validation.'),
  num_workers: L('데이터를 읽을 일꾼 프로세스 수예요. 윈도우에서 문제가 나면 0으로 두세요.', 'Worker processes for loading. Use 0 if Windows has trouble.'),
  momentum_sgd: L('이전 방향을 얼마나 이어갈지예요. 0.9가 흔해요.', 'How much of the previous update to keep. 0.9 is common.'),
  betas: L('Adam이 기울기 평균을 얼마나 오래 기억할지예요. 보통 그대로 둬요.', 'Decay rates for Adam moving averages. Usually left as is.'),
  shape: L('배치 크기를 뺀 입력 모양이에요. 예: (28, 28, 1)', 'Input shape without the batch size, e.g. (28, 28, 1).'),
  weight: L('클래스마다 loss에 줄 가중치예요. 데이터가 한쪽으로 치우쳤을 때 써요.', 'Per-class weights for imbalanced data.'),
  label_smoothing: L('정답을 100%가 아니라 조금 덜 확신하게 만들어 과적합을 줄여요.', 'Softens targets to reduce overconfidence.'),
  pos_weight: L('양성(1) 예제에 줄 가중치예요. 양성이 드물 때 키워요.', 'Weight for positive examples when they are rare.'),
};

const CONV_DIAGRAM = L(
  `입력 (N, C_in, H, W)
   │  kernel_size×kernel_size 창이 stride칸씩 이동
   ▼
출력 (N, C_out, H', W')
H' = (H + 2·padding − kernel_size) / stride + 1`,
  `input (N, C_in, H, W)
   │  a kernel_size×kernel_size window moves by stride
   ▼
output (N, C_out, H', W')
H' = (H + 2·padding − kernel_size) / stride + 1`,
);

const POOL_DIAGRAM = L(
  `┌──┬──┐
│ 1│ 3│      max   ┌──┐
├──┼──┤  ───────▶ │ 4│
│ 2│ 4│            └──┘
└──┴──┘   2×2 → 1칸`,
  `┌──┬──┐
│ 1│ 3│      max   ┌──┐
├──┼──┤  ───────▶ │ 4│
│ 2│ 4│            └──┘
└──┴──┘   2×2 → 1 cell`,
);

const LINEAR_DIAGRAM = L(`(N, in_features) ──▶ [ 가중치 곱 + 편향 ] ──▶ (N, out_features)`, `(N, in_features) ──▶ [ weights · x + bias ] ──▶ (N, out_features)`);

const FLATTEN_DIAGRAM = L(`(N, 64, 7, 7) ──펼치기──▶ (N, 64×7×7) = (N, 3136)`, `(N, 64, 7, 7) ──flatten──▶ (N, 64×7×7) = (N, 3136)`);

const DROPOUT_DIAGRAM = L(`학습 때:  ● ● ● ● ●  ──▶  ● ○ ● ○ ●   (○ = 꺼짐)
평가 때:  모두 사용`, `training: ● ● ● ● ●  ──▶  ● ○ ● ○ ●   (○ = dropped)
eval:     all kept`);

const RELU_DIAGRAM = L(`  출력
   │      ╱
   │    ╱
───┼──╱──── 입력
   │ (음수는 0)`, `  output
   │      ╱
   │    ╱
───┼──╱──── input
   │ (negatives → 0)`);

// ---- dictionary ----------------------------------------------------------------------

export const CONCEPTS: Record<string, Concept> = {
  'torch.Linear': {
    title: L('nn.Linear — 완전 연결 층', 'nn.Linear — fully connected layer'),
    what: {
      beginner: L(
        '들어온 숫자들을 모두 섞어서 새로운 숫자들을 만드는 층이에요. 이미지를 다룰 때는 보통 Flatten으로 한 줄로 펼친 다음에 써요. in_features는 바로 앞 출력의 마지막 크기와 꼭 같아야 해요.',
        'Mixes all input numbers into new numbers. For images, flatten first. in_features must equal the last size of the previous output.',
      ),
      normal: L(
        '입력 x에 대해 y = xWᵀ + b를 계산해요. 마지막 축에만 적용되므로 (N, *, in_features) 모양을 받아 (N, *, out_features)를 내보내요. 파라미터 수는 in_features×out_features + out_features예요.',
        'Computes y = xWᵀ + b on the last dimension: (N, *, in_features) → (N, *, out_features). Parameters: in×out + out.',
      ),
    },
    diagram: LINEAR_DIAGRAM,
    params: { in_features: P.in_features, out_features: P.out_features, bias: P.bias },
    doc: PT('Linear'),
  },
  'torch.Conv1d': {
    title: L('nn.Conv1d — 1차원 합성곱 층', 'nn.Conv1d — 1D convolution'),
    what: {
      beginner: L('소리나 시계열처럼 한 줄로 이어진 데이터를 작은 창으로 훑으며 특징을 찾는 층이에요.', 'Slides a small window along 1D data such as audio or time series to find patterns.'),
      normal: L('(N, C_in, L) 입력을 받아 (N, C_out, L′)을 내보내요. L′ = ⌊(L + 2·padding − dilation·(kernel_size−1) − 1)/stride⌋ + 1.', 'Maps (N, C_in, L) to (N, C_out, L′) with L′ = ⌊(L + 2p − d(k−1) − 1)/s⌋ + 1.'),
    },
    params: { in_channels: P.in_channels, out_channels: P.out_channels, kernel_size: P.kernel_size, stride: P.stride, padding: P.padding, dilation: P.dilation, groups: P.groups, bias: P.bias },
    doc: PT('Conv1d'),
  },
  'torch.Conv2d': {
    title: L('nn.Conv2d — 2차원 합성곱 층', 'nn.Conv2d — 2D convolution'),
    what: {
      beginner: L(
        '이미지 위를 작은 창(필터)으로 훑으면서 선, 모서리 같은 특징을 찾아내는 층이에요. 필터 하나가 채널 하나를 만들어요. in_channels는 들어오는 이미지의 채널 수와 같아야 해요.',
        'Slides small filters over an image to find features like edges. Each filter makes one channel. in_channels must match the input channels.',
      ),
      normal: L(
        '(N, C_in, H, W) 입력에 C_out개의 커널을 합성곱해 (N, C_out, H′, W′)를 만들어요. H′ = ⌊(H + 2·padding − dilation·(kernel_size−1) − 1)/stride⌋ + 1. 파라미터 수는 C_out·(C_in/groups)·k_h·k_w (+ C_out)예요.',
        'Convolves C_out kernels over (N, C_in, H, W) → (N, C_out, H′, W′); H′ = ⌊(H + 2p − d(k−1) − 1)/s⌋ + 1. Params: C_out·(C_in/groups)·k_h·k_w (+ C_out).',
      ),
    },
    diagram: CONV_DIAGRAM,
    params: { in_channels: P.in_channels, out_channels: P.out_channels, kernel_size: P.kernel_size, stride: P.stride, padding: P.padding, dilation: P.dilation, groups: P.groups, bias: P.bias },
    doc: PT('Conv2d'),
  },
  'torch.MaxPool2d': {
    title: L('nn.MaxPool2d — 최대 풀링', 'nn.MaxPool2d — max pooling'),
    what: {
      beginner: L('이미지를 작은 칸으로 나눠서 각 칸에서 가장 큰 값만 남겨 크기를 줄여요. 보통 2를 쓰면 가로·세로가 절반이 돼요. 학습할 가중치는 없어요.', 'Keeps the largest value in each small block to shrink the image. 2 halves height and width. No learnable weights.'),
      normal: L('채널 수는 그대로 두고 H, W만 줄여요. stride를 생략하면 kernel_size와 같아요. ceil_mode=True면 나머지 칸도 하나의 창으로 쳐요.', 'Reduces H and W, keeping channels. stride defaults to kernel_size. ceil_mode=True counts the leftover as a window.'),
    },
    diagram: POOL_DIAGRAM,
    params: { kernel_size: P.pool_kernel, stride: P.pool_stride, padding: P.padding, dilation: P.dilation, ceil_mode: P.ceil_mode },
    doc: PT('MaxPool2d'),
  },
  'torch.AvgPool2d': {
    title: L('nn.AvgPool2d — 평균 풀링', 'nn.AvgPool2d — average pooling'),
    what: {
      beginner: L('작은 칸마다 평균을 내서 이미지 크기를 줄여요. MaxPool보다 부드럽게 줄여요.', 'Averages each small block to shrink the image, smoother than max pooling.'),
      normal: L('창 안의 평균을 출력해요. 출력 크기 계산은 MaxPool2d와 같고 dilation은 없어요.', 'Outputs the window mean. Same size formula as MaxPool2d without dilation.'),
    },
    params: { kernel_size: P.pool_kernel, stride: P.pool_stride, padding: P.padding, ceil_mode: P.ceil_mode },
    doc: PT('AvgPool2d'),
  },
  'torch.AdaptiveAvgPool2d': {
    title: L('nn.AdaptiveAvgPool2d — 크기 맞춤 평균 풀링', 'nn.AdaptiveAvgPool2d — adaptive average pooling'),
    what: {
      beginner: L('입력 크기가 얼마든 원하는 크기로 맞춰 줘요. 1을 쓰면 채널마다 숫자 하나만 남아서 Linear에 넣기 편해요.', 'Shrinks any input to the size you want. 1 leaves one number per channel, handy before Linear.'),
      normal: L('출력이 항상 (N, C, *output_size)가 되도록 창 크기를 자동으로 정해 평균을 내요. ResNet의 마지막 부분에서 자주 써요.', 'Chooses windows automatically so the output is (N, C, *output_size). Common at the end of ResNets.'),
    },
    params: { output_size: P.output_size },
    doc: PT('AdaptiveAvgPool2d'),
  },
  'torch.Flatten': {
    title: L('nn.Flatten — 펼치기', 'nn.Flatten — flatten'),
    what: {
      beginner: L('여러 겹으로 쌓인 데이터를 한 줄로 쭉 펼쳐요. 합성곱 층 다음, Linear 층 앞에 넣어요. 배치 축은 그대로 남아요.', 'Lays stacked data out into one row. Put it after conv layers and before Linear. The batch dimension stays.'),
      normal: L('start_dim부터 end_dim까지의 축을 하나로 합쳐요. 기본값은 (N, C, H, W) → (N, C·H·W)예요. x.view(x.size(0), -1)과 같은 결과예요.', 'Merges dimensions start_dim..end_dim. Default (N, C, H, W) → (N, C·H·W), same as x.view(x.size(0), -1).'),
    },
    diagram: FLATTEN_DIAGRAM,
    params: { start_dim: P.start_dim, end_dim: P.end_dim },
    doc: PT('Flatten'),
  },
  'torch.BatchNorm1d': {
    title: L('nn.BatchNorm1d — 배치 정규화(1D)', 'nn.BatchNorm1d — batch normalization (1D)'),
    what: {
      beginner: L('층을 지난 숫자들이 너무 커지거나 작아지지 않게 고르게 맞춰 줘서 학습을 빠르고 안정적으로 만들어요. Linear 층 뒤에 자주 써요.', 'Keeps values in a steady range so training is faster and more stable. Often after Linear layers.'),
      normal: L('(N, C) 또는 (N, C, L) 입력의 채널마다 배치 평균·분산으로 정규화한 뒤 γ, β로 조정해요. 학습/평가 모드에 따라 동작이 달라서 model.eval()을 꼭 불러야 해요.', 'Normalizes each channel of (N, C) or (N, C, L) with batch statistics, then scales by γ and β. Behaves differently in train/eval; call model.eval().'),
    },
    params: { num_features: P.num_features, eps: P.eps, momentum: P.momentum },
    doc: PT('BatchNorm1d'),
  },
  'torch.BatchNorm2d': {
    title: L('nn.BatchNorm2d — 배치 정규화(2D)', 'nn.BatchNorm2d — batch normalization (2D)'),
    what: {
      beginner: L('합성곱 층을 지난 값들을 채널마다 고르게 맞춰 줘서 학습을 안정적으로 만들어요. 보통 Conv2d → BatchNorm2d → ReLU 순서로 써요.', 'Evens out values per channel after conv layers. Usually Conv2d → BatchNorm2d → ReLU.'),
      normal: L('(N, C, H, W) 입력의 채널마다 N·H·W 위치의 평균·분산으로 정규화해요. num_features는 C와 같아야 하고, 학습 파라미터는 2C개예요.', 'Normalizes each channel of (N, C, H, W) over N·H·W. num_features must equal C; 2C learnable parameters.'),
    },
    params: { num_features: P.num_features, eps: P.eps, momentum: P.momentum },
    doc: PT('BatchNorm2d'),
  },
  'torch.LayerNorm': {
    title: L('nn.LayerNorm — 층 정규화', 'nn.LayerNorm — layer normalization'),
    what: {
      beginner: L('데이터 하나하나의 마지막 숫자들을 고르게 맞춰요. 트랜스포머에서 자주 써요. 배치 크기와 상관없이 동작해요.', 'Evens out the last numbers of each sample. Common in Transformers; independent of batch size.'),
      normal: L('마지막 축(들)의 평균·분산으로 각 샘플을 정규화해요. BatchNorm과 달리 학습/평가에서 똑같이 동작해요.', 'Normalizes each sample over its last dimension(s). Same behavior in train and eval.'),
    },
    params: { normalized_shape: P.normalized_shape, eps: P.eps },
    doc: PT('LayerNorm'),
  },
  'torch.Dropout': {
    title: L('nn.Dropout — 드롭아웃', 'nn.Dropout — dropout'),
    what: {
      beginner: L('학습할 때 일부 숫자를 무작위로 꺼서 모델이 답을 외우지 못하게 해요(과적합 방지). 평가할 때는 꺼지지 않아요.', 'Randomly turns off some values during training so the model cannot just memorize. Off during evaluation.'),
      normal: L('학습 모드에서 확률 p로 원소를 0으로 만들고 나머지를 1/(1−p)배 해요. model.eval()에서는 그대로 통과해요. 크기는 바뀌지 않아요.', 'In training, zeros elements with probability p and scales others by 1/(1−p). Identity in eval. Shape unchanged.'),
    },
    diagram: DROPOUT_DIAGRAM,
    params: { p: P.dropout_p, inplace: P.inplace },
    doc: PT('Dropout'),
  },
  'torch.ReLU': {
    title: L('nn.ReLU — 렐루 활성화 함수', 'nn.ReLU — ReLU activation'),
    what: {
      beginner: L('음수는 0으로 바꾸고 양수는 그대로 두는 함수예요. 층과 층 사이에 넣어서 모델이 복잡한 모양도 배울 수 있게 해 줘요.', 'Turns negatives into 0 and keeps positives. Placed between layers so the model can learn complex shapes.'),
      normal: L('ReLU(x) = max(0, x). 계산이 빠르고 기울기 소실이 적어 가장 많이 써요. 크기는 바뀌지 않아요.', 'ReLU(x) = max(0, x). Fast and reduces vanishing gradients. Shape unchanged.'),
    },
    diagram: RELU_DIAGRAM,
    params: { inplace: P.inplace },
    doc: PT('ReLU'),
  },
  'torch.LeakyReLU': {
    title: L('nn.LeakyReLU — 리키 렐루', 'nn.LeakyReLU — leaky ReLU'),
    what: {
      beginner: L('ReLU와 비슷하지만 음수를 완전히 0으로 만들지 않고 아주 조금 남겨요. 뉴런이 아예 죽어 버리는 문제를 줄여 줘요.', 'Like ReLU but keeps a little of the negatives, so neurons do not die.'),
      normal: L('x ≥ 0이면 x, 아니면 negative_slope·x예요. GAN 등에서 자주 써요.', 'x if x ≥ 0 else negative_slope·x. Common in GANs.'),
    },
    params: { negative_slope: P.negative_slope, inplace: P.inplace },
    doc: PT('LeakyReLU'),
  },
  'torch.GELU': {
    title: L('nn.GELU — 젤루 활성화 함수', 'nn.GELU — GELU activation'),
    what: {
      beginner: L('ReLU를 부드럽게 만든 함수예요. 트랜스포머(BERT, GPT)에서 많이 써요.', 'A smooth version of ReLU, used in Transformers like BERT and GPT.'),
      normal: L('GELU(x) = x·Φ(x) (Φ는 표준정규분포의 누적분포함수). 크기는 바뀌지 않아요.', 'GELU(x) = x·Φ(x). Shape unchanged.'),
    },
    params: {},
    doc: PT('GELU'),
  },
  'torch.Sigmoid': {
    title: L('nn.Sigmoid — 시그모이드', 'nn.Sigmoid — sigmoid'),
    what: {
      beginner: L('어떤 숫자든 0과 1 사이로 바꿔요. "예/아니오" 문제의 마지막에 확률로 쓸 수 있어요. 손실로 BCELoss를 함께 써요.', 'Squashes any number into 0–1. Use at the end of yes/no problems with BCELoss.'),
      normal: L('σ(x) = 1/(1+e^(−x)). 출력층에서 쓸 때는 Sigmoid+BCELoss보다 BCEWithLogitsLoss 하나를 쓰는 편이 수치적으로 더 안정적이에요.', 'σ(x) = 1/(1+e^(−x)). Prefer BCEWithLogitsLoss over Sigmoid+BCELoss for stability.'),
    },
    params: {},
    doc: PT('Sigmoid'),
  },
  'torch.Tanh': {
    title: L('nn.Tanh — 하이퍼볼릭 탄젠트', 'nn.Tanh — tanh'),
    what: {
      beginner: L('어떤 숫자든 −1과 1 사이로 바꿔요. RNN 안에서 자주 쓰여요.', 'Squashes numbers into −1..1. Common inside RNNs.'),
      normal: L('tanh(x) = (eˣ−e^(−x))/(eˣ+e^(−x)). 출력이 0을 중심으로 대칭이에요.', 'Zero-centered squashing function.'),
    },
    params: {},
    doc: PT('Tanh'),
  },
  'torch.Softmax': {
    title: L('nn.Softmax — 소프트맥스', 'nn.Softmax — softmax'),
    what: {
      beginner: L('여러 숫자를 합이 1인 확률로 바꿔요. 분류에서 각 클래스일 확률을 볼 때 써요. 주의: CrossEntropyLoss를 쓸 때는 모델 끝에 Softmax를 넣지 마세요(안에 이미 들어 있어요).', 'Turns numbers into probabilities that sum to 1. Do not add it before CrossEntropyLoss (it is included).'),
      normal: L('dim 축을 따라 exp(xᵢ)/Σexp(xⱼ)를 계산해요. 학습에는 logits 그대로 CrossEntropyLoss에, 확률이 필요할 때만 Softmax를 써요.', 'exp(xᵢ)/Σexp(xⱼ) along dim. Train on logits with CrossEntropyLoss; use Softmax only for probabilities.'),
    },
    params: { dim: P.dim },
    doc: PT('Softmax'),
  },
  'torch.LogSoftmax': {
    title: L('nn.LogSoftmax — 로그 소프트맥스', 'nn.LogSoftmax — log softmax'),
    what: {
      beginner: L('소프트맥스 확률에 로그를 씌운 값이에요. NLLLoss와 짝으로 써요.', 'Log of softmax probabilities. Pair it with NLLLoss.'),
      normal: L('log(softmax(x))를 수치적으로 안정하게 계산해요. LogSoftmax + NLLLoss = CrossEntropyLoss예요.', 'Numerically stable log(softmax(x)). LogSoftmax + NLLLoss = CrossEntropyLoss.'),
    },
    params: { dim: P.dim },
    doc: PT('LogSoftmax'),
  },
  'torch.Embedding': {
    title: L('nn.Embedding — 임베딩(단어를 숫자 벡터로)', 'nn.Embedding — embedding'),
    what: {
      beginner: L('단어 번호(정수)를 숫자 묶음(벡터)으로 바꿔 주는 표예요. 입력은 꼭 정수(long)여야 해요.', 'A lookup table that turns token ids (integers) into vectors. Input must be integers (long).'),
      normal: L('(N, L) 정수 입력을 (N, L, embedding_dim)으로 바꿔요. 파라미터 수는 num_embeddings × embedding_dim이에요.', 'Maps (N, L) integer ids to (N, L, embedding_dim). Params: num_embeddings × embedding_dim.'),
    },
    params: { num_embeddings: P.num_embeddings, embedding_dim: P.embedding_dim },
    doc: PT('Embedding'),
  },
  'torch.LSTM': {
    title: L('nn.LSTM — 장단기 기억 신경망', 'nn.LSTM — long short-term memory'),
    what: {
      beginner: L('문장이나 시계열처럼 순서가 있는 데이터를 앞에서부터 차례로 읽으며 기억해요. 출력은 (output, (h, c)) 두 개라서 꺼내 쓰는 방법에 주의해야 해요.', 'Reads ordered data step by step and remembers. Returns (output, (h, c)) — unpack carefully.'),
      normal: L('batch_first=True면 (N, L, input_size) → output (N, L, D·hidden_size), h·c (D·num_layers, N, hidden_size). D는 양방향이면 2예요.', 'With batch_first=True: (N, L, input_size) → output (N, L, D·hidden), h/c (D·layers, N, hidden).'),
    },
    params: { input_size: P.input_size, hidden_size: P.hidden_size, num_layers: P.num_layers, batch_first: P.batch_first, bidirectional: P.bidirectional, dropout: P.dropout_p },
    doc: PT('LSTM'),
  },
  'torch.GRU': {
    title: L('nn.GRU — 게이트 순환 유닛', 'nn.GRU — gated recurrent unit'),
    what: {
      beginner: L('LSTM을 조금 단순하게 만든 순환 신경망이에요. 비슷한 성능에 더 가벼워요. 출력은 (output, h) 두 개예요.', 'A simpler LSTM with similar quality and fewer parameters. Returns (output, h).'),
      normal: L('리셋·업데이트 게이트 두 개만 써요. 모양 규칙은 LSTM과 같지만 셀 상태 c가 없어요.', 'Uses reset and update gates. Same shape rules as LSTM without the cell state.'),
    },
    params: { input_size: P.input_size, hidden_size: P.hidden_size, num_layers: P.num_layers, batch_first: P.batch_first, bidirectional: P.bidirectional },
    doc: PT('GRU'),
  },
  'torch.RNN': {
    title: L('nn.RNN — 기본 순환 신경망', 'nn.RNN — vanilla RNN'),
    what: {
      beginner: L('순서가 있는 데이터를 한 칸씩 읽으며 이전 기억을 넘겨주는 가장 기본적인 순환 신경망이에요. 긴 문장은 잘 기억하지 못해서 보통 LSTM이나 GRU를 써요.', 'The most basic recurrent network. Forgets long sequences, so LSTM/GRU are usually preferred.'),
      normal: L('hₜ = tanh(W·xₜ + U·hₜ₋₁ + b). 긴 시퀀스에서 기울기 소실이 심해요.', 'hₜ = tanh(W·xₜ + U·hₜ₋₁ + b). Suffers from vanishing gradients.'),
    },
    params: { input_size: P.input_size, hidden_size: P.hidden_size, num_layers: P.num_layers, batch_first: P.batch_first },
    doc: PT('RNN'),
  },
  'torch.Sequential': {
    title: L('nn.Sequential — 층을 순서대로 묶기', 'nn.Sequential — run layers in order'),
    what: {
      beginner: L('층들을 적은 순서대로 차례차례 통과시키는 묶음이에요. forward를 따로 쓰지 않아도 돼요.', 'Runs the listed layers one after another. No forward needed.'),
      normal: L('각 모듈의 출력이 다음 모듈의 입력이 돼요. 이름으로 접근하려면 OrderedDict를 넘겨요. 인덱스로 seq[0]처럼 꺼낼 수 있어요.', 'Each output feeds the next module. Pass an OrderedDict for named layers; index with seq[0].'),
    },
    params: {},
    doc: PT('Sequential'),
  },
  'torch.ModuleList': {
    title: L('nn.ModuleList — 층 목록', 'nn.ModuleList — list of layers'),
    what: {
      beginner: L('층들을 목록으로 담아 두는 상자예요. 그냥 파이썬 리스트에 담으면 PyTorch가 층을 못 찾아서 학습이 안 되니 꼭 ModuleList를 써요.', 'A list that PyTorch can see. A plain Python list hides layers from training — use ModuleList.'),
      normal: L('Sequential과 달리 자동으로 실행하지 않아요. forward에서 for 문으로 직접 꺼내 써요. 파라미터가 model.parameters()에 등록돼요.', 'Unlike Sequential it does not run automatically; iterate in forward. Parameters are registered.'),
    },
    params: {},
    doc: PT('ModuleList'),
  },
  'torch.CrossEntropyLoss': {
    title: L('nn.CrossEntropyLoss — 분류용 손실', 'nn.CrossEntropyLoss — classification loss'),
    what: {
      beginner: L('여러 개 중 하나를 고르는 분류 문제에 쓰는 손실이에요. 모델 출력은 Softmax 없이 그대로 넣고, 정답은 클래스 번호(정수)로 넣어요.', 'Loss for picking one class out of many. Feed raw model outputs (no Softmax) and integer class labels.'),
      normal: L('입력 (N, C) logits와 타깃 (N,) long을 받아요. 안에서 LogSoftmax + NLLLoss를 해요. 타깃이 float이면 dtype 에러가 나요.', 'Takes (N, C) logits and (N,) long targets; internally LogSoftmax + NLLLoss. Float targets cause a dtype error.'),
    },
    params: { weight: P.weight, reduction: P.reduction, label_smoothing: P.label_smoothing },
    doc: PT('CrossEntropyLoss'),
  },
  'torch.MSELoss': {
    title: L('nn.MSELoss — 평균 제곱 오차', 'nn.MSELoss — mean squared error'),
    what: {
      beginner: L('숫자를 맞히는 문제(회귀)에 쓰는 손실이에요. 예측과 정답의 차이를 제곱해서 평균 내요.', 'Loss for predicting numbers (regression): mean of squared differences.'),
      normal: L('mean((ŷ − y)²). 예측과 정답의 모양이 같아야 해요. (N, 1)과 (N,)을 섞으면 브로드캐스팅으로 잘못 계산될 수 있어요.', 'mean((ŷ − y)²). Shapes must match; mixing (N, 1) and (N,) silently broadcasts.'),
    },
    params: { reduction: P.reduction },
    doc: PT('MSELoss'),
  },
  'torch.BCELoss': {
    title: L('nn.BCELoss — 이진 분류 손실', 'nn.BCELoss — binary cross entropy'),
    what: {
      beginner: L('"예/아니오" 문제에 쓰는 손실이에요. 모델 끝에 Sigmoid를 넣어 0~1 확률을 만든 뒤 써요.', 'Loss for yes/no problems. Use after a Sigmoid that outputs 0–1.'),
      normal: L('입력이 반드시 0~1 범위여야 해요. 수치 안정성을 위해 BCEWithLogitsLoss를 추천해요.', 'Inputs must be in 0–1. BCEWithLogitsLoss is recommended for stability.'),
    },
    params: { weight: P.weight, reduction: P.reduction },
    doc: PT('BCELoss'),
  },
  'torch.BCEWithLogitsLoss': {
    title: L('nn.BCEWithLogitsLoss — 시그모이드가 들어간 이진 분류 손실', 'nn.BCEWithLogitsLoss — sigmoid + BCE'),
    what: {
      beginner: L('"예/아니오" 문제에 쓰는 손실이에요. 안에 Sigmoid가 들어 있으니 모델 끝에 Sigmoid를 따로 넣지 마세요.', 'Yes/no loss with Sigmoid built in — do not add another Sigmoid.'),
      normal: L('Sigmoid와 BCE를 합쳐 log-sum-exp로 안정하게 계산해요. 타깃은 float이어야 해요.', 'Combines Sigmoid and BCE stably. Targets must be float.'),
    },
    params: { pos_weight: P.pos_weight, reduction: P.reduction },
    doc: PT('BCEWithLogitsLoss'),
  },
  'torch.optim.SGD': {
    title: L('optim.SGD — 확률적 경사 하강법', 'optim.SGD — stochastic gradient descent'),
    what: {
      beginner: L('가장 기본적인 학습 방법이에요. 기울기 반대 방향으로 lr만큼 조금씩 고쳐요. momentum=0.9를 함께 쓰면 훨씬 잘 배워요.', 'The basic optimizer: step against the gradient by lr. momentum=0.9 helps a lot.'),
      normal: L('θ ← θ − lr·∇θ (momentum 사용 시 속도 항 추가). 학습률 스케줄러와 함께 쓰면 좋아요.', 'θ ← θ − lr·∇θ (with momentum term). Pairs well with LR schedulers.'),
    },
    params: { lr: P.lr, momentum: P.momentum_sgd, weight_decay: P.weight_decay },
    doc: 'https://pytorch.org/docs/stable/generated/torch.optim.SGD.html',
  },
  'torch.optim.Adam': {
    title: L('optim.Adam — 아담 옵티마이저', 'optim.Adam — Adam optimizer'),
    what: {
      beginner: L('가장 많이 쓰는 학습 방법이에요. 파라미터마다 알아서 고치는 크기를 조절해 줘요. 처음엔 lr=0.001(1e-3)로 시작해 보세요.', 'The most popular optimizer; adapts step size per parameter. Start with lr=0.001.'),
      normal: L('기울기의 1차·2차 이동평균으로 적응적 학습률을 써요. weight_decay를 쓸 거라면 AdamW가 더 올바르게 동작해요.', 'Adaptive learning rates from first/second moment estimates. Use AdamW for weight decay.'),
    },
    params: { lr: P.lr, betas: P.betas, weight_decay: P.weight_decay },
    doc: 'https://pytorch.org/docs/stable/generated/torch.optim.Adam.html',
  },
  'torch.optim.AdamW': {
    title: L('optim.AdamW — 가중치 감쇠를 고친 Adam', 'optim.AdamW — Adam with decoupled weight decay'),
    what: {
      beginner: L('Adam에 과적합을 줄이는 기능(weight_decay)을 제대로 붙인 버전이에요. 트랜스포머 학습에서 기본처럼 써요.', 'Adam with proper weight decay to fight overfitting. Default for Transformers.'),
      normal: L('가중치 감쇠를 기울기와 분리해서 적용해요. weight_decay 기본값은 0.01이에요.', 'Applies weight decay separately from the gradient. Default weight_decay 0.01.'),
    },
    params: { lr: P.lr, betas: P.betas, weight_decay: P.weight_decay },
    doc: 'https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html',
  },
  'torch.DataLoader': {
    title: L('DataLoader — 데이터를 배치로 나눠 주는 도구', 'DataLoader — batches your data'),
    what: {
      beginner: L('데이터셋을 batch_size만큼씩 묶어서 차례로 꺼내 줘요. 학습용은 shuffle=True로 섞어 주세요.', 'Hands out the dataset in batches of batch_size. Use shuffle=True for training.'),
      normal: L('Dataset을 받아 미니배치 이터레이터를 만들어요. 마지막 배치가 작으면 drop_last=True로 버릴 수 있어요. 윈도우에서는 num_workers>0일 때 if __name__ == "__main__": 보호가 필요해요.', 'Builds a mini-batch iterator. drop_last=True drops a small final batch. On Windows, num_workers>0 needs the __main__ guard.'),
    },
    params: { batch_size: P.batch_size, shuffle: P.shuffle, num_workers: P.num_workers },
    doc: 'https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader',
  },
  // ---- Keras ----
  'keras.Dense': {
    title: L('layers.Dense — 완전 연결 층', 'layers.Dense — fully connected layer'),
    what: {
      beginner: L('들어온 숫자들을 모두 섞어서 units개의 새 숫자를 만들어요. 분류 모델의 마지막 층이면 units를 클래스 수로 두고 activation="softmax"를 써요.', 'Mixes inputs into `units` new numbers. For the last classifier layer, set units to the number of classes with softmax.'),
      normal: L('마지막 축에만 적용돼요: (None, ..., d) → (None, ..., units). 파라미터 수는 d×units + units예요. 입력 크기는 처음 호출될 때 정해져요.', 'Applies to the last axis: (None, ..., d) → (None, ..., units). Params d×units + units. Built on first call.'),
    },
    diagram: LINEAR_DIAGRAM,
    params: { units: P.units, activation: P.activation, use_bias: P.bias, input_shape: P.input_shape },
    doc: KR('core_layers/dense'),
  },
  'keras.Conv2D': {
    title: L('layers.Conv2D — 2차원 합성곱 층', 'layers.Conv2D — 2D convolution'),
    what: {
      beginner: L('이미지 위를 작은 창으로 훑으며 특징을 찾아요. Keras는 채널이 맨 뒤에 와서 입력이 (배치, 높이, 너비, 채널) 모양이에요.', 'Finds features by sliding windows over images. Keras puts channels last: (batch, H, W, C).'),
      normal: L("padding='valid'면 H′ = ⌊(H − dilation·(k−1) − 1)/stride⌋ + 1, 'same'이면 ⌈H/stride⌉. 4차원 입력만 받아요.", "valid: H′ = ⌊(H − d(k−1) − 1)/s⌋ + 1; same: ⌈H/s⌉. Requires 4D input."),
    },
    diagram: CONV_DIAGRAM,
    params: { filters: P.filters, kernel_size: P.kernel_size, strides: P.strides, padding: P.k_padding, activation: P.activation, input_shape: P.input_shape },
    doc: KR('convolution_layers/convolution2d'),
  },
  'keras.MaxPooling2D': {
    title: L('layers.MaxPooling2D — 최대 풀링', 'layers.MaxPooling2D — max pooling'),
    what: {
      beginner: L('작은 칸마다 가장 큰 값만 남겨 이미지 크기를 줄여요. 기본값 2면 가로·세로가 절반이 돼요.', 'Keeps the max of each block to shrink images. Default 2 halves H and W.'),
      normal: L("strides를 생략하면 pool_size와 같아요. padding='valid'면 나머지 칸은 버려요.", "strides defaults to pool_size. 'valid' drops leftovers."),
    },
    diagram: POOL_DIAGRAM,
    params: { pool_size: P.pool_size, strides: P.strides, padding: P.k_padding },
    doc: KR('pooling_layers/max_pooling2d'),
  },
  'keras.Flatten': {
    title: L('layers.Flatten — 펼치기', 'layers.Flatten — flatten'),
    what: {
      beginner: L('여러 겹 데이터를 한 줄로 펼쳐요. Conv2D 다음, Dense 앞에 넣어요.', 'Flattens data into one row. Put it after Conv2D and before Dense.'),
      normal: L('배치 축을 빼고 나머지를 모두 곱해 하나로 합쳐요: (None, 5, 5, 64) → (None, 1600).', 'Merges all but the batch axis: (None, 5, 5, 64) → (None, 1600).'),
    },
    diagram: FLATTEN_DIAGRAM,
    params: {},
    doc: KR('reshaping_layers/flatten'),
  },
  'keras.Dropout': {
    title: L('layers.Dropout — 드롭아웃', 'layers.Dropout — dropout'),
    what: {
      beginner: L('학습할 때 일부 값을 무작위로 꺼서 모델이 답을 외우지 못하게 해요. 예측할 때는 꺼지지 않아요.', 'Randomly drops values during training to prevent memorizing. Off at prediction time.'),
      normal: L('training=True일 때만 rate 비율로 0을 만들고 나머지를 1/(1−rate)배 해요. 크기는 바뀌지 않아요.', 'Zeros a fraction `rate` only when training and rescales the rest. Shape unchanged.'),
    },
    diagram: DROPOUT_DIAGRAM,
    params: { rate: P.rate },
    doc: KR('regularization_layers/dropout'),
  },
  'keras.BatchNormalization': {
    title: L('layers.BatchNormalization — 배치 정규화', 'layers.BatchNormalization — batch normalization'),
    what: {
      beginner: L('층을 지난 값들을 고르게 맞춰서 학습을 빠르고 안정적으로 만들어요.', 'Evens out values so training is faster and more stable.'),
      normal: L('마지막 축(채널)마다 정규화해요. 파라미터는 4C개(학습 2C + 이동 통계 2C)로 표시돼요.', 'Normalizes per last-axis channel. Shows 4C params (2C trainable + 2C moving stats).'),
    },
    params: { momentum: P.momentum, epsilon: P.eps },
    doc: KR('normalization_layers/batch_normalization'),
  },
  'keras.Embedding': {
    title: L('layers.Embedding — 임베딩', 'layers.Embedding — embedding'),
    what: {
      beginner: L('단어 번호(정수)를 숫자 묶음(벡터)으로 바꿔 줘요.', 'Turns integer token ids into vectors.'),
      normal: L('(None, L) 정수 입력 → (None, L, output_dim). 파라미터 수는 input_dim × output_dim이에요.', '(None, L) ids → (None, L, output_dim). Params input_dim × output_dim.'),
    },
    params: { input_dim: P.num_embeddings, output_dim: P.embedding_dim },
    doc: KR('core_layers/embedding'),
  },
  'keras.LSTM': {
    title: L('layers.LSTM — 장단기 기억 신경망', 'layers.LSTM — long short-term memory'),
    what: {
      beginner: L('순서가 있는 데이터를 차례로 읽으며 기억해요. 기본은 마지막 시점의 출력만 돌려줘요. 층을 여러 개 쌓을 때는 return_sequences=True를 써요.', 'Reads sequences and remembers. Returns only the last output by default; use return_sequences=True to stack.'),
      normal: L('(None, L, d) → return_sequences=False면 (None, units), True면 (None, L, units).', '(None, L, d) → (None, units), or (None, L, units) with return_sequences=True.'),
    },
    params: { units: L('기억(은닉 상태)의 크기예요.', 'Size of the hidden state.'), return_sequences: L('모든 시점의 출력을 돌려줄지예요. LSTM을 여러 층 쌓을 때 True로 해요.', 'Return outputs for every step. True when stacking LSTMs.') },
    doc: KR('recurrent_layers/lstm'),
  },
  'keras.Input': {
    title: L('keras.Input — 입력 모양 정하기', 'keras.Input — define the input shape'),
    what: {
      beginner: L('모델에 들어갈 데이터 모양을 알려 줘요. 배치 크기는 빼고 적어요. 예: keras.Input(shape=(28, 28, 1))', 'Tells the model the input shape, without batch size, e.g. keras.Input(shape=(28, 28, 1)).'),
      normal: L('심볼릭 텐서를 만들어요. Sequential의 첫 원소로 넣거나 함수형 API의 시작점으로 써요.', 'Creates a symbolic tensor: first element of Sequential or start of the functional API.'),
    },
    params: { shape: P.shape, batch_size: P.batch_size },
    doc: 'https://keras.io/api/models/model/',
  },
  'keras.Sequential': {
    title: L('keras.Sequential — 층을 순서대로 쌓기', 'keras.Sequential — stack layers in order'),
    what: {
      beginner: L('층을 적은 순서대로 쌓아서 모델을 만들어요. 맨 앞에 keras.Input이나 input_shape를 주면 바로 모양을 알 수 있어요.', 'Builds a model by stacking layers in order. Give keras.Input or input_shape first to know the shapes.'),
      normal: L('입력과 출력이 하나씩인 단순한 모델에 써요. 분기나 여러 입력이 필요하면 함수형 API(keras.Model)를 써요.', 'For single-input/single-output stacks. Use the functional API for branches or multiple inputs.'),
    },
    params: {},
    doc: 'https://keras.io/api/models/sequential/',
  },
};

export const REQUIRED_CONCEPTS = [
  'Linear', 'Conv1d', 'Conv2d', 'MaxPool2d', 'AvgPool2d', 'AdaptiveAvgPool2d', 'Flatten', 'BatchNorm1d', 'BatchNorm2d',
  'LayerNorm', 'Dropout', 'ReLU', 'LeakyReLU', 'GELU', 'Sigmoid', 'Tanh', 'Softmax', 'LogSoftmax', 'Embedding', 'LSTM',
  'GRU', 'RNN', 'Sequential', 'ModuleList', 'CrossEntropyLoss', 'MSELoss', 'BCELoss', 'BCEWithLogitsLoss',
].map((n) => 'torch.' + n)
  .concat(['torch.optim.SGD', 'torch.optim.Adam', 'torch.optim.AdamW', 'torch.DataLoader'])
  .concat(['Dense', 'Conv2D', 'MaxPooling2D', 'Flatten', 'Dropout', 'BatchNormalization', 'Embedding', 'LSTM', 'Input', 'Sequential'].map((n) => 'keras.' + n));
