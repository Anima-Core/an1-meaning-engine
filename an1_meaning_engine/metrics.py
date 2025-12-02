"""Latency and FLOP estimation utilities."""

import time
import torch
import torch.nn as nn


def measure_latency(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_warmup: int = 10,
    num_samples: int = 1000,
) -> float:
    """Measure average latency per example in milliseconds."""
    model.eval()
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        # Warmup
        for i, (images, _) in enumerate(dataloader):
            if i >= num_warmup:
                break
            images = images.to(device, non_blocking=True)
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Actual timing
        for i, (images, _) in enumerate(dataloader):
            if total_samples >= num_samples:
                break
            images = images.to(device, non_blocking=True)
            start = time.perf_counter()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            batch_size = images.size(0)
            total_time += elapsed
            total_samples += batch_size

    if total_samples == 0:
        return 0.0
    return (total_time / total_samples) * 1000.0  # Convert to ms


def estimate_flops_an1(header_dim: int, num_classes: int = 10) -> float:
    """Estimate FLOPs for AN1 head (64 -> 1024 -> 512 -> 128 -> 10)."""
    # Each linear layer: 2 * (input_dim * output_dim) FLOPs (multiply-add)
    flops = 2.0 * (
        header_dim * 1024
        + 1024 * 512
        + 512 * 128
        + 128 * num_classes
    )
    return flops


def estimate_flops_resnet18() -> float:
    """Estimated FLOPs for ResNet18 on CIFAR-10 (32x32 input)."""
    return 1_800_000_000.0

