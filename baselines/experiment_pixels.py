import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PixelHead(nn.Module):
    """
    Simple MLP that sees raw pixels only.
    This mirrors the AN1 head size so we can compare fairly.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def get_loaders(batch_size: int = 256):
    """
    Matches the CIFAR-10 setup used in the main experiment.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def train_epoch(model, loader, opt, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_correct = 0
    total = 0

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    acc = total_correct / total
    latency_s_per_example = elapsed / total
    return acc, latency_s_per_example


def run_pixel_baseline(epochs: int = 20, batch_size: int = 256, lr: float = 1e-3):
    print(f"[Pixel] Device: {DEVICE}")
    train_loader, test_loader = get_loaders(batch_size)

    model = PixelHead().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        loss, acc = train_epoch(model, train_loader, opt, criterion)
        print(f"[Pixel] Epoch {epoch:03d} loss={loss:.4f} acc={acc * 100:.2f}%")

    test_acc, infer_time = evaluate(model, test_loader)
    print(f"[Pixel] Test accuracy: {test_acc * 100:.2f}%")
    print(f"[Pixel] Latency per example: {infer_time * 1000:.4f} ms")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/pixel_head.pth")

    return {
        "pixel_acc": float(test_acc),
        "pixel_latency_s_per_example": float(infer_time),
    }


if __name__ == "__main__":
    results = run_pixel_baseline()
    print("[Pixel] Results:", results)
