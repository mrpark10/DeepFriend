# 라벨이 데이터와 상관없이 뒤섞여서 정확도가 계속 찍기 수준(10%)인 예제예요.
import torch
import torch.nn as nn

import deepfriend
from _data import blobs, fit, loaders

deepfriend.watch(name="chance")
torch.manual_seed(0)
x, _ = blobs(n=3000, classes=10)
y = torch.randint(0, 10, (3000,))  # 섞인 라벨
train, val = loaders(x, y, n_train=2500, batch_size=64)
model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 10))
fit(model, train, val, torch.optim.Adam(model.parameters(), lr=1e-3), nn.CrossEntropyLoss(), epochs=5)
