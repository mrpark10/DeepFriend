# 정상적으로 잘 배우는 예제 — 학습 닥터가 아무 문제도 찾지 않아야 해요.
import torch
import torch.nn as nn

import deepfriend
from _data import blobs, fit, loaders

deepfriend.watch(name="healthy")
torch.manual_seed(0)
x, y = blobs(spread=2.0)
train, val = loaders(x, y, n_train=900, batch_size=32)
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 4))
fit(model, train, val, torch.optim.Adam(model.parameters(), lr=1e-3), nn.CrossEntropyLoss(), epochs=6)
