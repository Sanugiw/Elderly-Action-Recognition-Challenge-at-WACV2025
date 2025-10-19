import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
from pathlib import Path
import albumentations as A

from models.rgb_stream import RGBTransformer
from dataset_rgb import RGBDataset, get_video_paths

def get_transforms(mode='train'):
    """Get augmentation transforms"""
    if mode == 'train':
        return A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch in progress_bar:
        videos, labels, _ = batch
        videos = videos.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(videos)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / (progress_bar.n + 1),
            'acc': 100. * correct / total
        })
    
    return total_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        
        for batch in progress_bar:
            videos, labels, _ = batch
            videos = videos.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, _ = model(videos)
            loss = criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({
                'loss': total_loss / (progress_bar.n + 1),
                'acc': 100. * correct / total
            })
    
    return total_loss / len(dataloader), correct / total

def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project="ear-challenge",
            name=f"rgb-stream-{args.run_name}",
            config=vars(args)
        )
    
    # For this initial version, we'll use dummy data
    # In Day 2, you'll replace this with actual training data
    print("Note: Using test videos for initial testing")
    print("Replace with training data when available")
    
    video_paths = get_video_paths(args.data_dir)
    print(f"Found {len(video_paths)} videos")
    
    # Create dummy labels for testing (remove when you have real labels)
    dummy_labels = [i % 6 for i in range(len(video_paths))]
    
    # Split train/val
    split_idx = int(0.9 * len(video_paths))
    train_paths = video_paths[:split_idx]
    val_paths = video_paths[split_idx:]
    train_labels = dummy_labels[:split_idx]
    val_labels = dummy_labels[split_idx:]
    
    # Create datasets
    train_dataset = RGBDataset(
        train_paths,
        labels=train_labels,
        transform=get_transforms('train'),
        num_frames=args.num_frames
    )
    
    val_dataset = RGBDataset(
        val_paths,
        labels=val_labels,
        transform=get_transforms('val'),
        num_frames=args.num_frames
    )
    
    # Create dataloaders
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
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = RGBTransformer(
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        pretrained=True
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Training loop
    best_val_acc = 0.0
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch+1
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_dir / 'rgb_stream_best.pth')
            print(f"✓ Saved best model with val_acc: {val_acc:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_dir / f'rgb_stream_epoch_{epoch+1}.pth')
    
    print(f"\n🎉 Training completed! Best val accuracy: {best_val_acc:.4f}")
    
    if not args.no_wandb:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RGB Stream Model')
    parser.add_argument('--data_dir', type=str, default='data/videos',
                       help='Directory containing videos')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames per video')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--num_classes', type=int, default=6,
                       help='Number of classes')
    parser.add_argument('--run_name', type=str, default='v1',
                       help='Run name for wandb')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    
    args = parser.parse_args()
    main(args)