# 학습률이 너무 커서 loss가 NaN(숫자 아님)이 되는 예제예요.
import torch
import torch.nn as nn

import deepfriend
from _data import blobs, fit, loaders

deepfriend.watch(name="nan")
torch.manual_seed(0)
x, y = blobs()
train, val = loaders(x, y, n_train=900, batch_size=32)
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 4))
fit(model, train, val, torch.optim.SGD(model.parameters(), lr=500.0), nn.CrossEntropyLoss(), epochs=3)
