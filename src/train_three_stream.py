import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
from pathlib import Path
import numpy as np

from models.three_stream_model import ThreeStreamModel
from dataset_multimodal import create_dataloaders
from training_utils import (
    FocalLoss,
    LabelSmoothingLoss,
    MixUpAugmentation,
    mixup_criterion,
    WarmupCosineScheduler,
    EMA,
    AverageMeter,
    MetricTracker,
    save_checkpoint
)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, args, ema=None, mixup=None):
    """Train for one epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    # Track per-stream accuracy
    rgb_acc_meter = AverageMeter()
    pose_acc_meter = AverageMeter()
    flow_acc_meter = AverageMeter()
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(progress_bar):
        rgb, pose, angles, flow, labels, video_ids = batch
        
        # Move to device
        rgb = rgb.to(device)
        pose = pose.to(device)
        angles = angles.to(device)
        flow = flow.to(device)
        labels = labels.to(device)
        
        batch_size = rgb.size(0)
        
        # Apply MixUp if enabled
        if mixup and np.random.rand() < 0.5:
            rgb, pose, angles, flow, labels_a, labels_b, lam = mixup(
                rgb, pose, angles, flow, labels
            )
            use_mixup = True
        else:
            use_mixup = False
        
        # Forward pass
        optimizer.zero_grad()
        
        if args.track_stream_acc:
            logits, stream_outputs = model(rgb, pose, angles, flow, return_stream_outputs=True)
        else:
            logits = model(rgb, pose, angles, flow)
        
        # Compute loss
        if use_mixup:
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
        else:
            loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # Update EMA
        if ema:
            ema.update()
        
        # Calculate accuracy
        _, predicted = logits.max(1)
        
        if not use_mixup:
            correct = predicted.eq(labels).sum().item()
            accuracy = correct / batch_size
            acc_meter.update(accuracy, batch_size)
            
            # Track per-stream accuracy
            if args.track_stream_acc:
                _, rgb_pred = stream_outputs['rgb'].max(1)
                _, pose_pred = stream_outputs['pose'].max(1)
                _, flow_pred = stream_outputs['flow'].max(1)
                
                rgb_acc_meter.update(rgb_pred.eq(labels).sum().item() / batch_size, batch_size)
                pose_acc_meter.update(pose_pred.eq(labels).sum().item() / batch_size, batch_size)
                flow_acc_meter.update(flow_pred.eq(labels).sum().item() / batch_size, batch_size)
        
        # Update meters
        loss_meter.update(loss.item(), batch_size)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss_meter.avg,
            'acc': acc_meter.avg * 100
        })
    
    results = {
        'loss': loss_meter.avg,
        'acc': acc_meter.avg
    }
    
    if args.track_stream_acc:
        results.update({
            'rgb_acc': rgb_acc_meter.avg,
            'pose_acc': pose_acc_meter.avg,
            'flow_acc': flow_acc_meter.avg
        })
    
    return results


def validate(model, dataloader, criterion, device, epoch, args, ema=None):
    """Validate the model"""
    if ema:
        ema.apply_shadow()
    
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    # Track per-stream accuracy
    rgb_acc_meter = AverageMeter()
    pose_acc_meter = AverageMeter()
    flow_acc_meter = AverageMeter()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        
        for batch in progress_bar:
            rgb, pose, angles, flow, labels, video_ids = batch
            
            rgb = rgb.to(device)
            pose = pose.to(device)
            angles = angles.to(device)
            flow = flow.to(device)
            labels = labels.to(device)
            
            batch_size = rgb.size(0)
            
            # Forward pass
            if args.track_stream_acc:
                logits, stream_outputs = model(rgb, pose, angles, flow, return_stream_outputs=True)
            else:
                logits = model(rgb, pose, angles, flow)
            
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            _, predicted = logits.max(1)
            correct = predicted.eq(labels).sum().item()
            accuracy = correct / batch_size
            
            # Track per-stream accuracy
            if args.track_stream_acc:
                _, rgb_pred = stream_outputs['rgb'].max(1)
                _, pose_pred = stream_outputs['pose'].max(1)
                _, flow_pred = stream_outputs['flow'].max(1)
                
                rgb_acc_meter.update(rgb_pred.eq(labels).sum().item() / batch_size, batch_size)
                pose_acc_meter.update(pose_pred.eq(labels).sum().item() / batch_size, batch_size)
                flow_acc_meter.update(flow_pred.eq(labels).sum().item() / batch_size, batch_size)
            
            # Update meters
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(accuracy, batch_size)
            
            # Store predictions
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss_meter.avg,
                'acc': acc_meter.avg * 100
            })
    
    if ema:
        ema.restore()
    
    results = {
        'loss': loss_meter.avg,
        'acc': acc_meter.avg,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    if args.track_stream_acc:
        results.update({
            'rgb_acc': rgb_acc_meter.avg,
            'pose_acc': pose_acc_meter.avg,
            'flow_acc': flow_acc_meter.avg
        })
    
    return results


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project="ear-challenge",
            name=f"three-stream-{args.run_name}",
            config=vars(args)
        )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        video_dir=args.data_dir,
        pose_dir=args.pose_dir,
        flow_dir=args.flow_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        num_frames=args.num_frames
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating three-stream model...")
    model = ThreeStreamModel(
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        fusion_type=args.fusion_type,
        pretrained=True
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Loss function
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    elif args.loss_type == 'label_smooth':
        criterion = LabelSmoothingLoss(num_classes=args.num_classes, smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=args.min_lr
    )
    
    # EMA
    ema = EMA(model, decay=0.999) if args.use_ema else None
    
    # MixUp
    mixup = MixUpAugmentation(alpha=0.2) if args.use_mixup else None
    
    # Training loop
    best_val_acc = 0.0
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_results = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch+1, args, ema, mixup
        )
        
        # Validate
        val_results = validate(
            model, val_loader, criterion,
            device, epoch+1, args, ema
        )
        
        # Update learning rate
        current_lr = scheduler.step()
        
        # Print results
        print(f"\nTrain - Loss: {train_results['loss']:.4f}, Acc: {train_results['acc']:.4f}")
        print(f"Val   - Loss: {val_results['loss']:.4f}, Acc: {val_results['acc']:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        if args.track_stream_acc:
            print(f"\nPer-Stream Val Accuracy:")
            print(f"  RGB:  {val_results['rgb_acc']:.4f}")
            print(f"  Pose: {val_results['pose_acc']:.4f}")
            print(f"  Flow: {val_results['flow_acc']:.4f}")
        
        # Log to wandb
        if not args.no_wandb:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_results['loss'],
                'train_acc': train_results['acc'],
                'val_loss': val_results['loss'],
                'val_acc': val_results['acc'],
                'lr': current_lr
            }
            
            if args.track_stream_acc:
                log_dict.update({
                    'val_rgb_acc': val_results['rgb_acc'],
                    'val_pose_acc': val_results['pose_acc'],
                    'val_flow_acc': val_results['flow_acc']
                })
            
            wandb.log(log_dict)
        
        # Save checkpoint
        is_best = val_results['acc'] > best_val_acc
        if is_best:
            best_val_acc = val_results['acc']
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_results['acc'],
            'val_loss': val_results['loss'],
            'args': vars(args)
        }
        
        if ema:
            checkpoint['ema_shadow'] = ema.shadow
        
        save_checkpoint(
            checkpoint,
            checkpoint_dir / f'three_stream_epoch_{epoch+1}.pth',
            is_best=is_best
        )
        
        if is_best:
            print(f"✓ New best model! Val accuracy: {val_results['acc']:.4f}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                checkpoint,
                checkpoint_dir / f'three_stream_checkpoint_{epoch+1}.pth'
            )
    
    print("\n" + "=" * 60)
    print(f"🎉 Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("=" * 60)
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Three-Stream Model')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/videos')
    parser.add_argument('--pose_dir', type=str, default='data/poses')
    parser.add_argument('--flow_dir', type=str, default='data/optical_flow')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--val_split', type=float, default=0.1)
    
    # Model
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--fusion_type', type=str, default='attention',
                       choices=['attention', 'late', 'early'])
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adamw', 'sgd'])
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Loss
    parser.add_argument('--loss_type', type=str, default='ce',
                       choices=['ce', 'focal', 'label_smooth'])
    
    # Augmentation
    parser.add_argument('--use_mixup', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    
    # Others
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--run_name', type=str, default='v1')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--track_stream_acc', action='store_true')
    
    args = parser.parse_args()
    main(args)
    
      
