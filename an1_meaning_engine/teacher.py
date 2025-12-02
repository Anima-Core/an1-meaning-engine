"""ResNet18 teacher model for CIFAR-10."""

import torch
import torch.nn as nn
import torchvision
from pathlib import Path


def build_resnet18_teacher(num_classes: int = 10):
    """Build ResNet18 for CIFAR-10 (32x32 images)."""
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    return model


def load_teacher(
    checkpoint_path: str,
    device: torch.device,
    num_classes: int = 10,
):
    """
    Load a pretrained, frozen ResNet18 teacher from checkpoint.
    
    The teacher is ALWAYS loaded in eval mode with requires_grad=False.
    This function NEVER trains the teacher.
    
    Args:
        checkpoint_path: Path to the teacher checkpoint
        device: Device to load the model on
        num_classes: Number of classes (default: 10 for CIFAR-10)
    
    Returns:
        Frozen ResNet18 model in eval mode
    
    Raises:
        FileNotFoundError: If checkpoint does not exist
    """
    model = build_resnet18_teacher(num_classes=num_classes)
    checkpoint = Path(checkpoint_path)
    
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found at {checkpoint_path}. "
            "Please train the teacher first using scripts/train_teacher.py "
            "or provide a valid checkpoint path."
        )
    
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    return model


def extract_header(teacher: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Extract 64D intention header from early ResNet18 layer."""
    with torch.no_grad():
        x = teacher.conv1(images)
        x = teacher.bn1(x)
        x = teacher.relu(x)
        x = teacher.maxpool(x)
        x = teacher.layer1(x)
    # Global average pool to get 64D vector (64 channels from layer1)
    header = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    return header

