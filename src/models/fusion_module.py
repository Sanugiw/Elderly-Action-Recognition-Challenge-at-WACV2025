import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    """
    Multi-modal fusion with cross-attention
    Learns to combine RGB, Pose, and Flow features adaptively
    """
    def __init__(
        self,
        rgb_dim=768,
        pose_dim=640,
        flow_dim=1024,
        num_classes=6,
        fusion_dim=512
    ):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        
        # Project all modalities to common dimension
        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.flow_proj = nn.Sequential(
            nn.Linear(flow_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Multi-head self-attention for cross-modal interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Gating mechanism for adaptive weighting
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, 3),
            nn.Softmax(dim=1)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim // 2, num_classes)
        )
    
    def forward(self, rgb_features, pose_features, flow_features):
        """
        Args:
            rgb_features: (B, rgb_dim) from RGB stream
            pose_features: (B, pose_dim) from Pose stream
            flow_features: (B, flow_dim) from Flow stream
        
        Returns:
            logits: (B, num_classes)
        """
        B = rgb_features.size(0)
        
        # Project to common dimension
        rgb_proj = self.rgb_proj(rgb_features)      # (B, fusion_dim)
        pose_proj = self.pose_proj(pose_features)   # (B, fusion_dim)
        flow_proj = self.flow_proj(flow_features)   # (B, fusion_dim)
        
        # Stack for attention: (B, 3, fusion_dim)
        stacked = torch.stack([rgb_proj, pose_proj, flow_proj], dim=1)
        
        # Apply cross-attention
        attended, attention_weights = self.cross_attention(
            stacked, stacked, stacked
        )  # (B, 3, fusion_dim)
        
        # Compute adaptive weights using gating
        concat_features = torch.cat([rgb_proj, pose_proj, flow_proj], dim=1)
        gate_weights = self.gate(concat_features)  # (B, 3)
        gate_weights = gate_weights.unsqueeze(2)   # (B, 3, 1)
        
        # Weighted fusion
        fused = (attended * gate_weights).sum(dim=1)  # (B, fusion_dim)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits, gate_weights.squeeze(2)  # Return weights for analysis


class LateFusion(nn.Module):
    """
    Late fusion: average predictions from each stream
    Simple but effective baseline
    """
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, rgb_logits, pose_logits, flow_logits, weights=None):
        """
        Args:
            rgb_logits: (B, num_classes)
            pose_logits: (B, num_classes)
            flow_logits: (B, num_classes)
            weights: Optional (3,) tensor of weights
        
        Returns:
            fused_logits: (B, num_classes)
        """
        if weights is None:
            weights = torch.tensor([1/3, 1/3, 1/3], device=rgb_logits.device)
        
        # Weighted average of predictions
        fused = (
            weights[0] * rgb_logits +
            weights[1] * pose_logits +
            weights[2] * flow_logits
        )
        
        return fused


class EarlyFusion(nn.Module):
    """
    Early fusion: concatenate features before classification
    """
    def __init__(
        self,
        rgb_dim=768,
        pose_dim=640,
        flow_dim=1024,
        num_classes=6
    ):
        super().__init__()
        
        total_dim = rgb_dim + pose_dim + flow_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, num_classes)
        )
    
    def forward(self, rgb_features, pose_features, flow_features):
        """
        Args:
            rgb_features, pose_features, flow_features: Feature tensors
        
        Returns:
            logits: (B, num_classes)
        """
        # Concatenate all features
        combined = torch.cat([rgb_features, pose_features, flow_features], dim=1)
        
        # Classification
        logits = self.fusion(combined)
        
        return logits


# Test fusion modules
if __name__ == '__main__':
    # Test AttentionFusion
    fusion = AttentionFusion()
    
    rgb_feat = torch.randn(4, 768)
    pose_feat = torch.randn(4, 640)
    flow_feat = torch.randn(4, 1024)
    
    logits, weights = fusion(rgb_feat, pose_feat, flow_feat)
    print(f"AttentionFusion:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Gate weights shape: {weights.shape}")
    print(f"  Sample weights: {weights[0]}")
    
    # Test LateFusion
    late_fusion = LateFusion()
    rgb_logits = torch.randn(4, 6)
    pose_logits = torch.randn(4, 6)
    flow_logits = torch.randn(4, 6)
    
    fused = late_fusion(rgb_logits, pose_logits, flow_logits)
    print(f"\nLateFusion:")
    print(f"  Fused logits shape: {fused.shape}")
    
    # Test EarlyFusion
    early_fusion = EarlyFusion()
    logits = early_fusion(rgb_feat, pose_feat, flow_feat)
    print(f"\nEarlyFusion:")
    print(f"  Logits shape: {logits.shape}")
    
    print("\n✓ All fusion modules working!")
