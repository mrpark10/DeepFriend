import torch
import torch.nn as nn

# df: input (4, 10)
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        y = self.fc(x)
        return y
