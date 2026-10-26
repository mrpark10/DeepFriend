# 예제 2: 컬러 이미지(CIFAR-10 모양) 분류 — 합성곱 신경망(CNN)
# - Conv2d와 MaxPool2d를 지날 때마다 크기가 어떻게 바뀌는지 줄 끝에서 확인해 보세요.
# - nn.Conv2d 위에 마우스를 올리면 쉬운 설명이 나와요.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import deepfriend

deepfriend.watch(name="cifar-cnn")
torch.manual_seed(0)


# df: input (32, 3, 32, 32)
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def fake_cifar(n):
    labels = torch.randint(0, 10, (n,))
    images = torch.rand(n, 3, 32, 32) * 0.5
    for i, y in enumerate(labels):  # 클래스마다 색 채널과 위치가 다른 네모를 그려요
        c = int(y) % 3
        images[i, c, int(y) * 3 : int(y) * 3 + 4, 8:24] += 0.5
    return images, labels


x_train, y_train = fake_cifar(1500)
x_val, y_val = fake_cifar(300)
train = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
val = DataLoader(TensorDataset(x_val, y_val), batch_size=128)

model = SmallCNN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(2):
    model.train()
    for xb, yb in train:
        optimizer.zero_grad()
        loss_fn(model(xb), yb).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        for xb, yb in val:
            loss_fn(model(xb), yb)
    print(f"epoch {epoch + 1} 끝")
