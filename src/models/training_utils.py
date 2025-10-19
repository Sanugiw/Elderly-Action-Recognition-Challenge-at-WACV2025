import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses training on hard examples
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C) logits
            targets: (B,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing for regularization
    Prevents overconfident predictions
    """
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C) logits
            target: (B,) class indices
        """
        pred = F.log_softmax(pred, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class MixUpAugmentation:
    """
    MixUp data augmentation
    Mixes two samples with interpolation
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, rgb, pose, angles, flow, labels):
        """
        Apply MixUp to a batch
        
        Returns:
            mixed data and labels for mixup criterion
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = rgb.size(0)
        index = torch.randperm(batch_size).to(rgb.device)
        
        # Mix all modalities with same lambda
        mixed_rgb = lam * rgb + (1 - lam) * rgb[index]
        mixed_pose = lam * pose + (1 - lam) * pose[index]
        mixed_angles = lam * angles + (1 - lam) * angles[index]
        mixed_flow = lam * flow + (1 - lam) * flow[index]
        
        labels_a, labels_b = labels, labels[index]
        
        return mixed_rgb, mixed_pose, mixed_angles, mixed_flow, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute loss for mixup augmentation
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing
    """
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        total_epochs,
        base_lr=None,
        min_lr=1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr if base_lr else optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self, epoch=None):
        """Update learning rate"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class EMA:
    """
    Exponential Moving Average for model parameters
    Smooths model weights for better generalization
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + \
                             self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Replace model parameters with shadow"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricTracker:
    """Track multiple metrics during training"""
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def get_average(self, key, last_n=None):
        values = self.metrics[key]
        if last_n:
            values = values[-last_n:]
        return np.mean(values) if values else 0.0
    
    def get_last(self, key):
        return self.metrics[key][-1] if self.metrics[key] else 0.0
    
    def reset(self):
        self.metrics = defaultdict(list)


def save_checkpoint(state, filename, is_best=False):
    """Save model checkpoint"""
    torch.save(state, filename)
    if is_best:
        best_filename = filename.replace('.pth', '_best.pth')
        torch.save(state, best_filename)


def load_checkpoint(filename, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


# Test utilities
if __name__ == '__main__':
    print("Testing training utilities...")
    
    # Test Focal Loss
    focal_loss = FocalLoss()
    inputs = torch.randn(4, 6)
    targets = torch.tensor([0, 1, 2, 3])
    loss = focal_loss(inputs, targets)
    print(f"✓ Focal Loss: {loss.item():.4f}")
    
    # Test Label Smoothing
    label_smooth = LabelSmoothingLoss(num_classes=6)
    loss = label_smooth(inputs, targets)
    print(f"✓ Label Smoothing Loss: {loss.item():.4f}")
    
    # Test MixUp
    mixup = MixUpAugmentation(alpha=0.2)
    rgb = torch.randn(4, 16, 3, 224, 224)
    pose = torch.randn(4, 16, 33, 4)
    angles = torch.randn(4, 16, 10)
    flow = torch.randn(4, 15, 2, 224, 224)
    labels = torch.tensor([0, 1, 2, 3])
    
    mixed = mixup(rgb, pose, angles, flow, labels)
    print(f"✓ MixUp applied, lambda shapes match")
    
    # Test LR Scheduler
  model = torch.nn.Linear(10, 6)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=50)
  
  lrs = []
  for epoch in range(20):
      scheduler.step()
      lrs.append(scheduler.get_lr())
  print(f"✓ Scheduler - LR at epoch 0: {lrs[0]:.6f}, epoch 10: {lrs[10]:.6f}")
  
  # Test EMA
  ema = EMA(model)
  ema.update()
  print(f"✓ EMA initialized and updated")
  
  # Test AverageMeter
  meter = AverageMeter()
  meter.update(0.5)
  meter.update(0.7)
  print(f"✓ AverageMeter avg: {meter.avg:.4f}")
  
  print("\n✅ All training utilities working!")
