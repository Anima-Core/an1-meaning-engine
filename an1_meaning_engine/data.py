"""CIFAR-10 data loaders."""

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def get_loaders(batch_size: int = 256, test_batch_size: int = None, num_workers: int = 4):
    """
    Get CIFAR-10 train and test loaders.
    
    Args:
        batch_size: Batch size for training loader
        test_batch_size: Batch size for test loader (defaults to batch_size if not specified)
        num_workers: Number of data loading workers
    """
    if test_batch_size is None:
        test_batch_size = batch_size
    
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
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader

