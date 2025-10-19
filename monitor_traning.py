"""
Real-time training monitor
Checks training progress and sends alerts
"""
import time
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt

def check_training_progress():
    """Check current training status"""
    checkpoint_dir = Path('checkpoints')
    
    if not checkpoint_dir.exists():
        print("No checkpoints directory found")
        return
    
    # Find all checkpoints
    checkpoints = list(checkpoint_dir.glob('three_stream_*.pth'))
    
    if not checkpoints:
        print("No checkpoints found yet")
        return
    
    # Get latest checkpoint
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    print(f"\n{'='*60}")
    print(f"Training Progress Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Load checkpoint info
    import torch
    checkpoint = torch.load(latest, map_location='cpu')
    
    print(f"\nLatest Checkpoint: {latest.name}")
    print(f"Epoch: {checkpoint['epoch'] + 1}")
    print(f"Val Accuracy: {checkpoint['val_acc']:.4f}")
    print(f"Val Loss: {checkpoint['val_loss']:.4f}")
    
    # Find best checkpoint
    best_ckpt = checkpoint_dir / 'three_stream_epoch_*_best.pth'
    best_files = list(checkpoint_dir.glob('*_best.pth'))
    
    if best_files:
        best = torch.load(best_files[0], map_location='cpu')
        print(f"\nBest Val Accuracy: {best['val_acc']:.4f} (Epoch {best['epoch'] + 1})")
    
    # Disk usage
    total_size = sum(f.stat().st_size for f in checkpoints) / (1024**3)
    print(f"\nCheckpoint Storage: {total_size:.2f} GB")
    
    print(f"{'='*60}\n")


def plot_training_curves():
    """Plot training curves from WandB or logs"""
    # This would read from WandB API or log files
    print("Training curves available on WandB dashboard")


def cleanup_old_checkpoints(keep_last_n=5):
    """Remove old checkpoints to save space"""
    checkpoint_dir = Path('checkpoints')
    checkpoints = sorted(
        checkpoint_dir.glob('three_stream_epoch_*.pth'),
        key=lambda p: p.stat().st_mtime
    )
    
    # Keep best checkpoint and last N
    best_ckpt = list(checkpoint_dir.glob('*_best.pth'))
    to_keep = set(best_ckpt + checkpoints[-keep_last_n:])
    
    removed_count = 0
    for ckpt in checkpoints:
        if ckpt not in to_keep:
            ckpt.unlink()
            removed_count += 1
            print(f"Removed: {ckpt.name}")
    
    if removed_count:
        print(f"\nCleaned up {removed_count} old checkpoints")
    else:
        print("No checkpoints to clean up")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up old checkpoints')
    parser.add_argument('--keep', type=int, default=5,
                       help='Number of recent checkpoints to keep')
    parser.add_argument('--watch', action='store_true',
                       help='Continuously monitor training')
    parser.add_argument('--interval', type=int, default=300,
                       help='Check interval in seconds (default: 5 minutes)')
    
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_old_checkpoints(keep_last_n=args.keep)
    elif args.watch:
        print("Starting training monitor...")
        print(f"Checking every {args.interval} seconds (Ctrl+C to stop)")
        try:
            while True:
                check_training_progress()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
    else:
        check_training_progress()
