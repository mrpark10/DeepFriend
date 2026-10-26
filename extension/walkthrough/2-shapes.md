## 줄마다 데이터 크기 보기 · Shapes on every line

모델 클래스 위에 한 줄만 적으세요 · Add one line above your model class:

```python
# df: input (32, 3, 224, 224)
class Net(nn.Module):
```

레이어 줄 끝에 `→ (32, 64, 112, 112)` 같은 회색 글씨가 나타나요. 크기가 안 맞으면 빨간 밑줄과 빠른 수정 버튼이 생겨요.
Each layer line shows its output shape. Mismatches get a red squiggle and a quick fix.
