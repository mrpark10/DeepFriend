# 예제 4: 일부러 넣은 크기 에러 — 실행하기 전에 DeepFriend가 먼저 알려줘요
# 1) BrokenNet의 nn.Linear(10000, 10) 아래 빨간 밑줄에 마우스를 올려 보세요.
#    "앞 레이어 출력은 12544개인데 Linear는 10000개를 기대해요" 라고 알려주고,
#    전구(빠른 수정)를 누르면 12544로 고쳐 줘요.
# 2) 고치기 전에 그냥 실행해 보면(터미널에서 python 04_shape_error.py)
#    영어 에러 대신 한국어 해설 카드가 떠요.
# 3) NoInputYet 위의 "입력 크기 정하기"를 누르면 그 모델의 크기도 볼 수 있어요.
import torch
import torch.nn as nn
import torch.nn.functional as F


# df: input (8, 1, 28, 28)
class BrokenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(10000, 10)  # ← 여기가 틀렸어요

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


class NoInputYet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.flatten(1))


model = BrokenNet()
out = model(torch.zeros(8, 1, 28, 28))
print(out.shape)
