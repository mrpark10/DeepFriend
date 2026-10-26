import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, cin=64, cout=64, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.short(x)
        return F.relu(out)


# df: input (2, 3, 32, 32)
class TinyResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.layer1 = BasicBlock(64, 64, 1)
        self.layer2 = BasicBlock(64, 128, 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def double_conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


# df: input (1, 1, 64, 64)
class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)
        self.dec = nn.Conv2d(48, 16, 3, padding=1)
        self.out = nn.Conv2d(16, 2, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        u = self.up(d2)
        cat = torch.cat([u, d1], dim=1)
        y = F.relu(self.dec(cat))
        return self.out(y)


# df: input (4, 3, 32, 32)
class VGGLoop(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1)])
        self.head = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        for conv in self.blocks:
            x = F.max_pool2d(F.relu(conv(x)), 2)
        x = x.reshape(x.shape[0], -1)
        return self.head(x)


# df: input (8, 50)
class TextCNN(nn.Module):
    def __init__(self, vocab=1000, dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.conv = nn.Conv1d(dim, 64, kernel_size=5)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(64, 2)

    def forward(self, tokens):
        e = self.emb(tokens)
        e = e.transpose(1, 2)
        h = F.relu(self.conv(e))
        h = F.adaptive_max_pool1d(h, 1).squeeze(-1)
        return self.fc(self.drop(h))


# df: input (16, 784)
class AutoEncoder(nn.Module):
    def __init__(self, latent=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())

    def forward(self, x):
        z = self.enc(x)
        recon = self.dec(z)
        loss_term = (recon - x).pow(2).mean(dim=1, keepdim=True)
        return recon, loss_term


# df: input (2, 10, 32)
class WithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.norm = nn.LayerNorm(32)
        self.fc = nn.Linear(32, 5)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        h = self.norm(x + a)
        return self.fc(h.mean(dim=1))
