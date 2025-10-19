import torch
import torch.nn as nn
import timm

class RGBTransformer(nn.Module):
    """
    Video Transformer for RGB appearance features
    Uses ViT backbone with temporal modeling
    """
    def __init__(self, num_classes=6, num_frames=16, embed_dim=768, pretrained=True):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        
        # Load pretrained Vision Transformer backbone
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Temporal positional embeddings
        self.temporal_embed = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim)
        )
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        
        # Temporal transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=12,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - Batch, Time, Channels, Height, Width
        
        Returns:
            logits: (B, num_classes)
            features: (B, embed_dim) - for fusion
        """
        B, T, C, H, W = x.shape
        
        # Flatten temporal dimension to process all frames
        x = x.view(B * T, C, H, W)
        
        # Extract spatial features with ViT
        spatial_features = self.vit.forward_features(x)  # (B*T, N+1, D)
        
        # Take CLS token from each frame
        cls_tokens = spatial_features[:, 0, :]  # (B*T, D)
        
        # Reshape to (B, T, D)
        temporal_features = cls_tokens.view(B, T, self.embed_dim)
        
        # Add temporal positional embeddings
        temporal_features = temporal_features + self.temporal_embed
        
        # Apply temporal transformer
        temporal_features = self.temporal_transformer(temporal_features)  # (B, T, D)
        
        # Global average pooling over time
        pooled_features = temporal_features.mean(dim=1)  # (B, D)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return logits, pooled_features

# Test the model
if __name__ == '__main__':
    model = RGBTransformer(num_classes=6, num_frames=16)
    
    # Test input
    x = torch.randn(2, 16, 3, 224, 224)  # Batch=2, Frames=16
    
    logits, features = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")