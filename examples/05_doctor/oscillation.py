# 학습률은 크고 배치는 너무 작아서 loss가 위아래로 크게 출렁이는 예제예요.
import torch
import torch.nn as nn

import deepfriend
from _data import blobs, fit, loaders

deepfriend.watch(name="oscillation")
torch.manual_seed(0)
x, y = blobs(spread=2.5)
train, val = loaders(x, y, n_train=300, batch_size=2)
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 4))
fit(model, train, val, torch.optim.SGD(model.parameters(), lr=0.1), nn.CrossEntropyLoss(), epochs=2)
