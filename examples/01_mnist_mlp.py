# 예제 1: 손글씨 숫자(MNIST 모양) 분류 — 가장 간단한 MLP
# - 인터넷 없이 돌아가도록 28×28 크기의 가짜 이미지를 만들어서 써요.
# - 줄 끝의 회색 글씨(→ (64, 128))가 DeepFriend의 "크기 표시"예요.
# - 실행하면 아래 패널의 "학습 닥터"에 그래프가 나타나요.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import deepfriend

deepfriend.watch(name="mnist-mlp")
torch.manual_seed(0)


# df: input (64, 1, 28, 28)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def fake_mnist(n):
    labels = torch.randint(0, 10, (n,))
    images = torch.rand(n, 1, 28, 28) * 0.3
    for i, y in enumerate(labels):  # 숫자마다 다른 위치에 밝은 줄을 그려서 배울 거리를 줘요
        images[i, 0, 2 + int(y) * 2 : 4 + int(y) * 2, 4:24] += 0.7
    return images, labels


x_train, y_train = fake_mnist(2000)
x_val, y_val = fake_mnist(500)
train = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
val = DataLoader(TensorDataset(x_val, y_val), batch_size=256)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for xb, yb in train:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in val:
            out = model(xb)
            loss_fn(out, yb)
            correct += (out.argmax(1) == yb).sum().item()
    print(f"epoch {epoch + 1}: 검증 정확도 {correct / len(x_val):.1%}")
