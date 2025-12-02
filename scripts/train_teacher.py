#!/usr/bin/env python3
"""
Train ResNet18 teacher on CIFAR-10.

This script trains the teacher ONCE and saves it to checkpoints/resnet18_cifar10_teacher.pth.
The main experiment (experiment_frozen_sender.py) will load this frozen checkpoint.
"""

import sys
from pathlib import Path

# Add parent directory to path to import from an1_meaning_engine
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

from an1_meaning_engine.teacher import build_resnet18_teacher
from an1_meaning_engine.data import CIFAR_MEAN, CIFAR_STD


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loaders(batch_size: int = 256, num_workers: int = 4):
    """Get CIFAR-10 train and test loaders."""
    transform_train = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    transform_test = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate model accuracy on dataloader."""
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100.0 * correct / total


def main():
    """Train ResNet18 teacher on CIFAR-10."""
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_loaders()

    model = build_resnet18_teacher(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

    epochs = 120
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_epoch = time.perf_counter()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        elapsed = time.perf_counter() - start_epoch
        print(
            f"[Teacher] Epoch {epoch:03d} "
            f"loss={train_loss:.4f} "
            f"acc={train_acc:.2f}% "
            f"time={elapsed:.2f}s"
        )

    test_acc = evaluate(model, test_loader, device)
    print(f"[Teacher] Final test accuracy: {test_acc:.2f}%")

    ckpt_path = Path("checkpoints")
    ckpt_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = ckpt_path / "resnet18_cifar10_teacher.pth"
    torch.save(model.state_dict(), checkpoint_file)
    print(f"Saved teacher checkpoint to {checkpoint_file}")


if __name__ == "__main__":
    main()

