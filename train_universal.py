"""
Training script for Universal Shape Recognition
"""

import os
import sys
import yaml
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.universal_dataset import UniversalDataset, get_dataloaders
from models.pointnet2_sem_seg import PointNet2SemSeg
from core.config_loader import UniversalConfig


def load_config(config_path: str) -> UniversalConfig:
    """Load configuration from YAML file."""
    return UniversalConfig(config_path)


def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Compute mean IoU."""
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    grad_clip: float = 5.0
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for points, labels in pbar:
        points = points.to(device)  # (B, N, 3)
        labels = labels.to(device)  # (B, N)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(points)  # (B, num_classes, N)
        predictions = predictions.transpose(1, 2)  # (B, N, num_classes)

        # Compute loss
        loss = criterion(predictions.reshape(-1, predictions.size(-1)), labels.reshape(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Compute accuracy
        pred_labels = torch.argmax(predictions, dim=2)
        correct = (pred_labels == labels).sum().item()
        total_correct += correct
        total_samples += labels.numel()
        total_loss += loss.item() * points.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int
) -> Tuple[float, float, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc='Validating')
    for points, labels in pbar:
        points = points.to(device)
        labels = labels.to(device)

        predictions = model(points)
        predictions = predictions.transpose(1, 2)

        loss = criterion(predictions.reshape(-1, predictions.size(-1)), labels.reshape(-1))

        pred_labels = torch.argmax(predictions, dim=2)
        correct = (pred_labels == labels).sum().item()
        total_correct += correct
        total_samples += labels.numel()
        total_loss += loss.item() * points.size(0)

        all_preds.append(pred_labels.cpu())
        all_targets.append(labels.cpu())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_samples

    # Compute mIoU
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    miou = compute_miou(all_preds.view(-1), all_targets.view(-1), num_classes)

    return avg_loss, accuracy, miou


def main():
    parser = argparse.ArgumentParser(description='Train Universal Shape Recognition Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output', type=str, default='models/universal', help='Output directory')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    train_config = config.training

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Create dataloaders
    print('Loading datasets...')
    train_loader, val_loader = get_dataloaders(
        args.data,
        batch_size=train_config['batch_size'],
        num_points=8192,
        num_workers=4
    )
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')

    # Create model
    model = PointNet2SemSeg(num_classes=train_config['num_classes']).to(device)
    print(f'Model created with {train_config["num_classes"]} classes')

    # Loss function with class weights
    class_weights = torch.FloatTensor(train_config.get('class_weights', [1.0] * train_config['num_classes'])).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    if train_config.get('optimizer', 'adam') == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=train_config['learning_rate'], momentum=0.9, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler_type = train_config.get('scheduler', 'step')
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_config['epochs'], eta_min=1e-6
        )
    else:  # step
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_config.get('step_size', 20),
            gamma=train_config.get('gamma', 0.5)
        )

    # Tensorboard
    writer = SummaryWriter(os.path.join(args.output, 'logs'))

    # Training loop
    best_miou = 0.0
    best_accuracy = 0.0

    print(f'Starting training for {train_config["epochs"]} epochs...')
    for epoch in range(1, train_config['epochs'] + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            grad_clip=train_config.get('grad_clip', 5.0)
        )

        # Validate
        val_loss, val_acc, val_miou = validate(model, val_loader, criterion, device, train_config['num_classes'])

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log
        print(f'Epoch {epoch}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val mIoU: {val_miou:.4f}')
        print(f'  Learning Rate: {current_lr:.6f}')

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('mIoU/Val', val_miou, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            best_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'miou': val_miou,
                'accuracy': val_acc,
            }, os.path.join(args.output, 'best_model.pth'))
            print(f'  ★ New best model saved! mIoU: {val_miou:.4f}')

    print(f'\nTraining complete!')
    print(f'Best validation accuracy: {best_accuracy:.4f}')
    print(f'Best validation mIoU: {best_miou:.4f}')
    print(f'Models saved to: {args.output}')

    writer.close()


if __name__ == '__main__':
    main()
