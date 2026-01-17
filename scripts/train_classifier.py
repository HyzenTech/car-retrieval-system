"""
Training script for the Car Type Classifier.

Usage:
    python scripts/train_classifier.py --epochs 30 --batch_size 32
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from classifier import CarTypeClassifier, CAR_TYPES, NUM_CLASSES
from utils import AverageMeter, ensure_dir, get_device

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Training will continue without logging.")


def get_args():
    parser = argparse.ArgumentParser(description='Train Car Type Classifier')
    
    # Data
    parser.add_argument('--data_dir', type=str, 
                        default='dataset/dataset',
                        help='Path to dataset directory')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Model
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='Freeze backbone weights')
    parser.add_argument('--unfreeze_epoch', type=int, default=5,
                        help='Epoch to unfreeze backbone (if frozen)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='models/classifier',
                        help='Output directory for checkpoints')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def get_transforms(img_size: int, train: bool = True):
    """Get data transforms."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        
        # Update meters
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc.item(), batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })
    
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        
        # Store predictions
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update meters
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc.item(), batch_size)
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })
    
    return loss_meter.avg, acc_meter.avg, np.array(all_preds), np.array(all_labels)


def main():
    args = get_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = get_device()
    print(f"Using device: {device}")
    
    # Setup output directory
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_dir = Path(args.output_dir) / args.exp_name
    ensure_dir(output_dir)
    print(f"Output directory: {output_dir}")
    
    # TensorBoard
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=output_dir / 'logs')
    else:
        writer = None
    
    # Data
    print("\nLoading data...")
    data_dir = Path(args.data_dir)
    
    train_transform = get_transforms(args.img_size, train=True)
    val_transform = get_transforms(args.img_size, train=False)
    
    train_dataset = datasets.ImageFolder(data_dir / 'train', transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / 'val', transform=val_transform)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    print("\nCreating model...")
    model = CarTypeClassifier(
        num_classes=NUM_CLASSES,
        pretrained=args.pretrained,
        dropout=args.dropout
    )
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        print("Freezing backbone weights...")
        for param in model.features.parameters():
            param.requires_grad = False
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone at specified epoch
        if args.freeze_backbone and epoch == args.unfreeze_epoch:
            print(f"\nUnfreezing backbone at epoch {epoch}")
            for param in model.features.parameters():
                param.requires_grad = True
            
            # Update optimizer with all parameters
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr * 0.1,  # Lower LR for fine-tuning
                weight_decay=args.weight_decay
            )
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=args.epochs - epoch + 1, 
                eta_min=1e-6
            )
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LR', current_lr, epoch)
        
        # Print summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"  New best model saved! Val Acc: {val_acc:.4f}")
        
        # Save latest model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'args': vars(args)
        }
        torch.save(checkpoint, output_dir / 'latest_model.pth')
    
    # Training complete
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print("="*50)
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
