"""
Universal training script for multi-shape recognition.
Configuration-driven training with support for multiple shape types.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent))

from core.config_loader import load_config
from models.pointnet2_sem_seg import get_model


class UniversalPointCloudDataset(Dataset):
    """Dataset for universal shape recognition."""

    def __init__(self, data_dir: str, num_points: int = 4096):
        self.data_dir = Path(data_dir)
        self.num_points = num_points
        self.files = sorted(list(self.data_dir.glob('scene_*.npz')))

        if len(self.files) == 0:
            raise ValueError(f"No data files found in {data_dir}")

        print(f"Loaded {len(self.files)} scenes from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        points = data['points']
        normals = data['normals']
        labels = data['labels']

        # Sample points
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
        else:
            indices = np.random.choice(len(points), self.num_points, replace=True)

        points = points[indices]
        normals = normals[indices]
        labels = labels[indices]

        # Normalize points
        center = points.mean(axis=0)
        points = points - center

        # Combine features
        features = np.concatenate([points, normals], axis=1)

        return torch.FloatTensor(features), torch.LongTensor(labels)


def train_epoch(model, dataloader, optimizer, criterion, device, num_classes):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for features, labels in pbar:
        features = features.transpose(2, 1).to(device)  # (B, 6, N)
        labels = labels.to(device)  # (B, N)

        optimizer.zero_grad()
        outputs = model(features)  # (B, N, num_classes) - 模型实际输出格式

        # Reshape for loss: (B, N, C) -> (B*N, C) and (B, N) -> (B*N)
        outputs_flat = outputs.contiguous().view(-1, num_classes)
        labels_flat = labels.view(-1)

        loss = criterion(outputs_flat, labels_flat)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy - outputs is (B, N, C)
        preds = outputs.argmax(dim=2).view(-1)  # (B*N,)
        labels_flat_check = labels.view(-1)  # (B*N,)
        correct += (preds == labels_flat_check).sum().item()
        total += labels_flat_check.numel()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device, num_classes):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Per-class metrics
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for features, labels in pbar:
            features = features.transpose(2, 1).to(device)
            labels = labels.to(device)

            outputs = model(features)

            # Reshape for loss - outputs is (B, N, C)
            outputs_flat = outputs.contiguous().view(-1, num_classes)
            labels_flat = labels.view(-1)
            loss = criterion(outputs_flat, labels_flat)

            total_loss += loss.item()

            # Compute accuracy
            preds = outputs.argmax(dim=2).view(-1)  # (B*N,)
            labels_flat = labels.view(-1)
            correct += (preds == labels_flat).sum().item()
            total += labels_flat.numel()

            # Per-class accuracy
            for c in range(num_classes):
                mask = labels_flat == c
                if mask.sum() > 0:
                    class_correct[c] += (preds[mask] == labels_flat[mask]).sum().item()
                    class_total[c] += mask.sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    # Compute per-class accuracy
    class_acc = class_correct / (class_total + 1e-10)

    return total_loss / len(dataloader), correct / total, class_acc


def main():
    parser = argparse.ArgumentParser(description='Train universal shape recognition model')
    parser.add_argument('--config', type=str, default='config/shape_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/universal',
                       help='Data directory containing train/ and test/ subdirectories')
    parser.add_argument('--output', type=str, default='models/universal',
                       help='Output directory for model checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--step_size', type=int, default=None,
                       help='Step size for scheduler (overrides config)')
    parser.add_argument('--gamma', type=float, default=None,
                       help='Gamma for scheduler (overrides config)')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override config with command-line arguments if provided
    if args.epochs is not None:
        config.training['epochs'] = args.epochs
        print(f"Overriding epochs: {args.epochs}")
    if args.batch_size is not None:
        config.training['batch_size'] = args.batch_size
        print(f"Overriding batch_size: {args.batch_size}")
    if args.learning_rate is not None:
        config.training['learning_rate'] = args.learning_rate
        print(f"Overriding learning_rate: {args.learning_rate}")
    if args.step_size is not None:
        config.training['step_size'] = args.step_size
        print(f"Overriding step_size: {args.step_size}")
    if args.gamma is not None:
        config.training['gamma'] = args.gamma
        print(f"Overriding gamma: {args.gamma}")

    # Setup device
    use_gpu = config.training.get('use_gpu', True)
    device = torch.device('cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Create datasets
    train_dataset = UniversalPointCloudDataset(
        Path(args.data) / 'train',
        num_points=config.inference['num_points']
    )
    test_dataset = UniversalPointCloudDataset(
        Path(args.data) / 'test',
        num_points=config.inference['num_points']
    )

    # Create dataloaders
    num_workers = config.training.get('num_workers', 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )

    # Create model
    model = get_model(config.num_classes).to(device)
    print(f"Model created with {config.num_classes} classes")

    # Load pretrained weights if specified
    pretrained_path = config.training.get('pretrained_weights', None)
    if pretrained_path and Path(pretrained_path).exists():
        print(f"Loading pretrained weights from {pretrained_path}")
        pretrained = torch.load(pretrained_path, map_location=device)
        if 'model_state_dict' in pretrained:
            pretrained = pretrained['model_state_dict']

        # Load weights with flexible matching
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained weights")

    # Loss function with class weights
    class_weights = torch.FloatTensor(config.get_class_weights()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer_type = config.training.get('optimizer', 'adam').lower()
    weight_decay = config.training.get('weight_decay', 0.0)

    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training['learning_rate'],
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.training['learning_rate'],
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:  # adam
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training['learning_rate'],
            weight_decay=weight_decay
        )

    print(f"Optimizer: {optimizer_type}, LR: {config.training['learning_rate']}, Weight Decay: {weight_decay}")

    # Learning rate scheduler
    scheduler_type = config.training.get('scheduler', 'step').lower()

    if scheduler_type == 'cosine':
        warmup_epochs = config.training.get('warmup_epochs', 0)
        min_lr = config.training.get('min_lr', 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training['epochs'] - warmup_epochs,
            eta_min=min_lr
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
    else:  # step
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.get('step_size', 20),
            gamma=config.training.get('gamma', 0.5)
        )

    print(f"Scheduler: {scheduler_type}")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.4f}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_class_acc': []
    }

    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    for epoch in range(start_epoch, config.training['epochs']):
        print(f"\nEpoch {epoch+1}/{config.training['epochs']}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, config.num_classes
        )

        # Validate
        val_loss, val_acc, class_acc = validate(
            model, test_loader, criterion, device, config.num_classes
        )

        # Update scheduler
        if scheduler_type == 'plateau':
            scheduler.step(val_acc)
        else:
            scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Per-class accuracy:")
        for label_id, shape_name in config.label_to_shape.items():
            print(f"    {shape_name}: {class_acc[label_id]:.4f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_class_acc'].append(class_acc.tolist())

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_acc': best_acc,
            'config_path': args.config
        }

        # Save latest
        torch.save(checkpoint, output_dir / 'latest_model.pth')

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint['best_acc'] = best_acc
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"  [BEST] New best model saved! Accuracy: {best_acc:.4f}")

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print("[SUCCESS] TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
