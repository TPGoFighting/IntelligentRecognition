"""
Universal Dataset for Shape Recognition
Supports 6 classes: 0=background, 1=unlabeled, 2=cylinder, 3=sphere, 4=cuboid, 5=plane
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional


class UniversalDataset(Dataset):
    """Dataset for universal shape recognition (cylinder, sphere, cuboid, plane)."""

    NUM_CLASSES = 6
    CLASS_NAMES = ['background', 'unlabeled', 'cylinder', 'sphere', 'cuboid', 'plane']

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        num_points: int = 8192,
        augment: bool = False
    ):
        """
        Args:
            data_root: Root directory of the dataset
            split: 'train', 'val', or 'test'
            num_points: Number of points to sample from each scene
            augment: Whether to apply data augmentation
        """
        self.data_root = data_root
        self.split = split
        self.num_points = num_points
        self.augment = augment

        # Load file list
        self.file_list = self._load_file_list()

    def _load_file_list(self) -> List[str]:
        """Load list of files for current split."""
        split_dir = os.path.join(self.data_root, self.split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        files = [f for f in os.listdir(split_dir) if f.endswith('.npz')]
        if not files:
            raise ValueError(f"No .npz files found in {split_dir}")

        return sorted(files)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            points: (N, 3) tensor of point coordinates
            labels: (N,) tensor of class labels
        """
        file_path = os.path.join(self.data_root, self.split, self.file_list[idx])
        data = np.load(file_path)

        points = data['points']  # (N, 3)
        labels = data['labels']  # (N,)

        # Sample points if too many
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
            labels = labels[indices]
        elif len(points) < self.num_points:
            # Pad with duplicates if too few
            indices = np.random.choice(len(points), self.num_points - len(points), replace=True)
            points = np.vstack([points, points[indices]])
            labels = np.concatenate([labels, labels[indices]])

        # Data augmentation
        if self.augment:
            points, labels = self._augment(points, labels)

        # Convert to tensors
        points = torch.FloatTensor(points)  # (N, 3)
        labels = torch.LongTensor(labels)  # (N,)

        return points, labels

    def _augment(self, points: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Random rotation around Z axis
        angle = np.random.uniform(-np.pi, np.pi)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        points = points @ rotation_matrix.T

        # Random scaling
        scale = np.random.uniform(0.8, 1.2)
        points = points * scale

        # Random translation
        translation = np.random.uniform(-0.5, 0.5, size=3)
        points = points + translation

        # Gaussian noise
        noise = np.random.normal(0, 0.01, size=points.shape)
        points = points + noise

        return points, labels


def get_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_points: int = 8192,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Returns:
        train_loader, val_loader
    """
    train_dataset = UniversalDataset(data_root, split='train', num_points=num_points, augment=True)
    val_dataset = UniversalDataset(data_root, split='val', num_points=num_points, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset
    import sys
    if len(sys.argv) < 2:
        print("Usage: python universal_dataset.py <data_root>")
        sys.exit(1)

    data_root = sys.argv[1]
    dataset = UniversalDataset(data_root, split='train', num_points=8192, augment=True)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Classes: {UniversalDataset.CLASS_NAMES}")

    points, labels = dataset[0]
    print(f"Sample points shape: {points.shape}")
    print(f"Sample labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels.numpy(), minlength=6)}")
