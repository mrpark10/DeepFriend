# 학습률이 너무 작아서 loss가 거의 움직이지 않는 예제예요.
import torch
import torch.nn as nn

import deepfriend
from _data import blobs, fit, loaders

deepfriend.watch(name="plateau")
torch.manual_seed(0)
x, y = blobs()
train, val = loaders(x, y, n_train=900, batch_size=32)
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 4))
fit(model, train, val, torch.optim.SGD(model.parameters(), lr=1e-7), nn.CrossEntropyLoss(), epochs=4)
