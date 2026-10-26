# 데이터는 적은데 모델이 너무 커서 답을 외워 버리는(과적합) 예제예요.
import torch
import torch.nn as nn

import deepfriend
from _data import blobs, fit, loaders

deepfriend.watch(name="overfit")
torch.manual_seed(0)
x, y = blobs(n=1040, classes=4, dim=20, spread=6.0)
train, val = loaders(x, y, n_train=40, batch_size=8)
model = nn.Sequential(nn.Linear(20, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 4))
fit(model, train, val, torch.optim.Adam(model.parameters(), lr=1e-3), nn.CrossEntropyLoss(), epochs=40)
