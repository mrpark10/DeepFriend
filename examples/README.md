# DeepFriend 예제 / Examples

| 파일 | 무엇을 볼 수 있나요 |
|---|---|
| `01_mnist_mlp.py` | 가장 간단한 모델, 줄마다 크기 표시, 학습 닥터 그래프 |
| `02_cifar_cnn.py` | 합성곱(CNN)에서 크기가 줄어드는 모습, 개념 설명 호버 |
| `03_keras_cnn.py` | Keras 모델 크기 표시, `watch()` 자동 연결 |
| `04_shape_error.py` | 실행 전에 크기 에러 찾기 + 빠른 수정, 한국어 에러 해설 |
| `05_doctor/*.py` | 학습 닥터가 진단하는 5가지 문제(NaN, 과적합, 정체, 출렁임, 찍기 수준)와 정상 학습 |

모든 예제는 인터넷 없이 가짜(합성) 데이터로 CPU에서 1분 안에 끝나요.
All examples use synthetic data and finish on a CPU in about a minute, offline.

실행: VS Code 터미널에서 `python 01_mnist_mlp.py` (PyTorch 필요), `python 03_keras_cnn.py` (TensorFlow 필요).
