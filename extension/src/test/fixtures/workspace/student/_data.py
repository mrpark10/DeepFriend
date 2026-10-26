"""Small synthetic datasets so every example runs offline on a CPU in seconds."""
import torch
from torch.utils.data import DataLoader, TensorDataset


def blobs(n=1200, classes=4, dim=10, spread=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(classes, dim, generator=g) * 3
    y = torch.randint(0, classes, (n,), generator=g)
    x = centers[y] + torch.randn(n, dim, generator=g) * spread
    return x, y


def loaders(x, y, n_train, batch_size, seed=0):
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(TensorDataset(x[:n_train], y[:n_train]), batch_size=batch_size, shuffle=True, generator=g)
    val = DataLoader(TensorDataset(x[n_train:], y[n_train:]), batch_size=256)
    return train, val


def fit(model, train, val, opt, loss_fn, epochs):
    for _ in range(epochs):
        model.train()
        for xb, yb in train:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            for xb, yb in val:
                loss_fn(model(xb), yb)
