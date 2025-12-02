"""Main experiment: AN1 Meaning Engine – Foreign Frozen Sender on CIFAR-10."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from .data import get_loaders
from .teacher import load_teacher, extract_header
from .an1_head import AN1Head
from .metrics import estimate_flops_an1, estimate_flops_resnet18


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Experiment configuration
TEACHER_CHECKPOINT = Path("checkpoints/resnet18_cifar10_teacher.pth")
BATCH_SIZE = 256  # Training batch size
TEST_BATCH_SIZE = 32  # Test/evaluation batch size (smaller for more dramatic speedups)
BENCH_BATCH_SIZE = 256  # Benchmark batch size for latency measurements
NUM_EPOCHS = 80
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 10


def evaluate_model(model: nn.Module, dataloader, device: torch.device) -> float:
    """Evaluate model accuracy on dataloader using full ResNet18 on images."""
    model.eval()
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)
    return 100.0 * total_correct / total_seen


def create_headers_loader(teacher: nn.Module, image_loader, device: torch.device, batch_size: int):
    """Precompute headers from images and create a DataLoader for headers + labels."""
    teacher.eval()
    all_headers = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in image_loader:
            images = images.to(device, non_blocking=True)
            headers = extract_header(teacher, images)
            all_headers.append(headers.cpu())
            all_labels.append(labels.cpu())
    
    headers_tensor = torch.cat(all_headers, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    headers_dataset = TensorDataset(headers_tensor, labels_tensor)
    headers_loader = DataLoader(
        headers_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # No need for workers since data is already in memory
        pin_memory=True,
    )
    return headers_loader


def evaluate_an1(an1_head: nn.Module, headers_loader, device: torch.device) -> float:
    """Evaluate AN1 head accuracy on precomputed 64-dim headers (no teacher call)."""
    an1_head.eval()
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for headers, labels in headers_loader:
            headers = headers.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = an1_head(headers)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)
    return 100.0 * total_correct / total_seen


def train_an1_head(
    an1_head: nn.Module,
    teacher: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
):
    """Train AN1 head to mimic teacher's logits."""
    # Precompute headers for validation evaluation
    test_headers_loader = create_headers_loader(teacher, test_loader, device, batch_size=TEST_BATCH_SIZE)
    
    optimizer = AdamW(an1_head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "an1_head_best.pth"

    for epoch in range(1, num_epochs + 1):
        an1_head.train()
        running_loss = 0.0
        total_correct = 0
        total_seen = 0
        start_epoch = time.perf_counter()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                teacher_logits = teacher(images)
                headers = extract_header(teacher, images)
            # Add small noise for regularization during training
            if an1_head.training:
                headers = headers + 0.01 * torch.randn_like(headers)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = an1_head(headers)
                # Knowledge distillation loss
                log_s = F.log_softmax(logits / 2.0, dim=1)
                log_t = F.softmax(teacher_logits / 2.0, dim=1)
                kd_loss = F.kl_div(log_s, log_t, reduction="batchmean") * (2.0 * 2.0)
                ce_loss = criterion(logits, labels)
                loss = 0.6 * kd_loss + 0.4 * ce_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)

        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start_epoch
        avg_loss = running_loss / total_seen
        train_acc = 100.0 * total_correct / total_seen
        print(
            f"[AN1] Epoch {epoch:03d} "
            f"loss={avg_loss:.4f} acc={train_acc:.2f}% time={epoch_time:.2f}s"
        )

        val_acc = evaluate_an1(an1_head, test_headers_loader, device)
        print(f"[AN1] Val acc={val_acc:.2f}% best={best_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(an1_head.state_dict(), best_ckpt_path)
            print(f"[AN1] Saved best checkpoint to {best_ckpt_path}")

    # Load best checkpoint
    if best_ckpt_path.exists():
        an1_head.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"[AN1] Loaded best checkpoint from {best_ckpt_path}")


def measure_teacher_latency(
    teacher: nn.Module,
    dataloader,
    device: torch.device,
    num_warmup: int = 10,
    num_timed_batches: int = 50,
) -> float:
    """Measure teacher latency using full ResNet18 on images."""
    teacher.eval()
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        # Warmup
        warmup_count = 0
        for images, _ in dataloader:
            if warmup_count >= num_warmup:
                break
            images = images.to(device, non_blocking=True)
            _ = teacher(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            warmup_count += 1

        # Actual timing - full teacher on images
        timed_count = 0
        for images, _ in dataloader:
            if timed_count >= num_timed_batches:
                break
            images = images.to(device, non_blocking=True)
            start = time.perf_counter()
            _ = teacher(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            batch_size_actual = images.size(0)
            total_time += elapsed
            total_samples += batch_size_actual
            timed_count += 1

    if total_samples == 0:
        return 0.0
    return (total_time / total_samples) * 1000.0  # Convert to ms per example


def measure_an1_latency(
    an1_head: nn.Module,
    headers_loader,
    device: torch.device,
    num_warmup: int = 10,
    num_timed_batches: int = 50,
) -> float:
    """Measure AN1 head latency using precomputed headers (no teacher call)."""
    an1_head.eval()
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        # Warmup
        warmup_count = 0
        for headers, _ in headers_loader:
            if warmup_count >= num_warmup:
                break
            headers = headers.to(device, non_blocking=True)
            _ = an1_head(headers)
            if device.type == "cuda":
                torch.cuda.synchronize()
            warmup_count += 1

        # Actual timing - AN1 head only on headers
        timed_count = 0
        for headers, _ in headers_loader:
            if timed_count >= num_timed_batches:
                break
            headers = headers.to(device, non_blocking=True)
            start = time.perf_counter()
            _ = an1_head(headers)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            batch_size_actual = headers.size(0)
            total_time += elapsed
            total_samples += batch_size_actual
            timed_count += 1

    if total_samples == 0:
        return 0.0
    return (total_time / total_samples) * 1000.0  # Convert to ms per example


def main():
    """Run the frozen sender experiment."""
    set_seed(42)
    
    parser = argparse.ArgumentParser(
        description="AN1 Meaning Engine – Foreign Frozen Sender on CIFAR-10"
    )
    parser.add_argument(
        "--one-pass",
        dest="one_pass",
        action="store_true",
        help="Use analytic one-pass AN1 training (Linear layer, no epoch loop).",
    )
    parser.add_argument(
        "--one-pass-only",
        dest="one_pass",
        action="store_true",
        help="Alias for --one-pass. Use analytic one-pass AN1 training (Linear layer, no epoch loop).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Get data loaders
    print("\n[1/4] Loading CIFAR-10 data...")
    print(f"Training batch size: {BATCH_SIZE}, Test batch size: {TEST_BATCH_SIZE}")
    train_loader, test_loader = get_loaders(batch_size=BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE)

    # Load frozen teacher (NEVER trains)
    print("[2/4] Loading frozen ResNet18 teacher...")
    teacher = load_teacher(
        str(TEACHER_CHECKPOINT),
        device,
        num_classes=NUM_CLASSES,
    )
    
    # Verify teacher is frozen
    is_frozen = all(not p.requires_grad for p in teacher.parameters())
    is_eval = not teacher.training
    print(f"Teacher frozen: {is_frozen and is_eval}")
    if not (is_frozen and is_eval):
        raise RuntimeError("Teacher must be frozen and in eval mode!")

    if args.one_pass:
        # One-pass mode: closed-form solve
        print("\n" + "=" * 60)
        print("ONE-PASS MODE: Analytic solve (no epoch loop)")
        print("=" * 60)
        
        # Collect headers and logits
        print("[OnePass] Collecting headers and teacher logits...")
        headers = []
        logits = []
        teacher.eval()
        with torch.no_grad():
            for images, _ in train_loader:
                images = images.to(device, non_blocking=True)
                h = extract_header(teacher, images)
                out = teacher(images)
                headers.append(h.detach().cpu())
                logits.append(out.detach().cpu())
        H = torch.cat(headers, dim=0)  # [N, 64]
        T = torch.cat(logits, dim=0)   # [N, 10]
        
        # Solve ridge regression
        print("[OnePass] Solving ridge regression...")
        H = H.to(device)
        T = T.to(device)
        HT = H.t()
        HTH = HT @ H
        HTH_reg = HTH + 1e-3 * torch.eye(HTH.shape[0], device=device)
        HTT = HT @ T
        W = torch.linalg.solve(HTH_reg, HTT)  # [64, 10]
        
        # Create linear head
        head = nn.Linear(64, 10, bias=True).to(device)
        with torch.no_grad():
            head.weight.copy_(W.t())
            head.bias.zero_()
        
        # Evaluate
        head.eval()
        correct = 0
        total = 0
        start_time = time.perf_counter()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                h = extract_header(teacher, images)
                logits = head(h)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        elapsed = time.perf_counter() - start_time
        acc = 100.0 * correct / total
        latency_ms = (elapsed / total) * 1000.0
        
        # Print results
        print("\n" + "=" * 60)
        print("AN1 Meaning Engine – ONE PASS Results")
        print("=" * 60)
        print(f"AN1 one-pass accuracy              : {acc:.2f}%")
        print(f"AN1 one-pass latency per example   : {latency_ms:.4f} ms")
        print("=" * 60)
        return

    # Default mode: Multi-epoch MLP training
    # Evaluate teacher on full ResNet18 with images
    print("[3/4] Evaluating teacher (full ResNet18 on images)...")
    teacher_acc = evaluate_model(teacher, test_loader, device)
    print(f"Teacher accuracy: {teacher_acc:.2f}%")

    # Determine header dimension
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 32, 32, device=device)
        header_dim = extract_header(teacher, dummy).shape[1]
    print(f"Header dimension: {header_dim}")

    # Precompute headers for AN1 evaluation and benchmarking
    print("[Precompute] Extracting headers from test set...")
    test_headers_loader = create_headers_loader(teacher, test_loader, device, batch_size=BENCH_BATCH_SIZE)

    # Create and train AN1 head
    print(f"[4/4] Training AN1 head ({header_dim}D -> {NUM_CLASSES} classes)...")
    an1_head = AN1Head(in_dim=header_dim, num_classes=NUM_CLASSES).to(device)
    train_an1_head(an1_head, teacher, train_loader, test_loader, device)

    # Final evaluation - AN1 on precomputed headers only
    print("\n[Final] Evaluating AN1 head (on precomputed headers)...")
    an1_acc = evaluate_an1(an1_head, test_headers_loader, device)
    print(f"AN1 accuracy: {an1_acc:.2f}%")

    # Measure latencies
    print("\n[Benchmark] Measuring latencies...")
    print(f"Using batch size: {BENCH_BATCH_SIZE}")
    
    # Create benchmark loaders with BENCH_BATCH_SIZE
    _, bench_test_loader = get_loaders(batch_size=BENCH_BATCH_SIZE, test_batch_size=BENCH_BATCH_SIZE)
    bench_headers_loader = create_headers_loader(teacher, bench_test_loader, device, batch_size=BENCH_BATCH_SIZE)
    
    # Measure teacher latency (full ResNet18 on images)
    teacher_latency_ms = measure_teacher_latency(teacher, bench_test_loader, device)
    
    # Measure AN1 latency (AN1 head only on precomputed headers)
    an1_latency_ms = measure_an1_latency(an1_head, bench_headers_loader, device)

    # Calculate metrics
    speedup = teacher_latency_ms / an1_latency_ms if an1_latency_ms > 0 else float("inf")
    resnet_flops = estimate_flops_resnet18()
    an1_flops = estimate_flops_an1(header_dim, NUM_CLASSES)
    flop_reduction = resnet_flops / an1_flops

    # Optional: Measure at different batch sizes
    print("\n[Benchmark] Measuring speedup at different batch sizes...")
    batch_sizes = [64, 256]
    speedup_table = []
    
    for bs in batch_sizes:
        _, bs_test_loader = get_loaders(batch_size=bs, test_batch_size=bs)
        bs_headers_loader = create_headers_loader(teacher, bs_test_loader, device, batch_size=bs)
        
        teacher_lat_bs = measure_teacher_latency(teacher, bs_test_loader, device)
        an1_lat_bs = measure_an1_latency(an1_head, bs_headers_loader, device)
        speedup_bs = teacher_lat_bs / an1_lat_bs if an1_lat_bs > 0 else float("inf")
        
        speedup_table.append({
            "batch_size": bs,
            "teacher_ms": teacher_lat_bs,
            "an1_ms": an1_lat_bs,
            "speedup": speedup_bs
        })

    # Print results table
    print("\n" + "=" * 60)
    print("AN1 Meaning Engine – Foreign Frozen Sender on CIFAR-10")
    print("=" * 60)
    print(f"Teacher accuracy                    : {teacher_acc:.2f}%")
    print(f"AN1 accuracy                        : {an1_acc:.2f}%")
    print(f"Teacher latency per example (ms)    : {teacher_latency_ms:.4f}")
    print(f"AN1 latency per example (ms)        : {an1_latency_ms:.4f}")
    print(f"Speedup (x)                         : {speedup:.2f}")
    print(f"FLOP reduction (x)                  : {flop_reduction:.1f}")
    print("=" * 60)
    
    # Print batch size comparison table
    if speedup_table:
        print("\nSpeedup by Batch Size:")
        print("-" * 60)
        print(f"{'Batch Size':<12} {'Teacher (ms)':<15} {'AN1 (ms)':<15} {'Speedup (x)':<15}")
        print("-" * 60)
        for row in speedup_table:
            print(f"{row['batch_size']:<12} {row['teacher_ms']:<15.4f} {row['an1_ms']:<15.4f} {row['speedup']:<15.2f}")
        print("-" * 60)
    
    print("\nNote: Teacher latency measured using full ResNet18 on images.")
    print("      AN1 latency measured using only AN1 head on precomputed 64-dim headers.")


if __name__ == "__main__":
    main()

