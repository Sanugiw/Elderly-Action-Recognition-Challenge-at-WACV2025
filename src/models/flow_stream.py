import torch
import torch.nn as nn
import timm

class OpticalFlowCNN(nn.Module):
    """
    CNN + LSTM for optical flow motion features
    Captures body motion direction and speed
    """
    def __init__(self, num_classes=6, num_frames=15):  # 15 flow frames (16-1)
        super().__init__()
        self.num_frames = num_frames
        
        # CNN backbone for spatial feature extraction
        # Use EfficientNet with modified input (2 channels for flow x,y)
        self.backbone = timm.create_model(
            'efficientnet_b2',
            pretrained=True,
            in_chans=2,  # Optical flow has 2 channels
            num_classes=0,  # Remove classifier
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 2, 224, 224)
            feat = self.backbone(dummy)
            self.feat_dim = feat.shape[1]
            self.feat_h = feat.shape[2]
            self.feat_w = feat.shape[3]
        
        # Spatial pooling
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Temporal modeling with Bi-LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Attention mechanism for temporal features
        self.temporal_attention = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, flow):
        """
        Args:
            flow: (B, T, 2, H, W) - optical flow
        
        Returns:
            logits: (B, num_classes)
            features: (B, 1024) - for fusion
        """
        B, T, C, H, W = flow.shape
        
        # Process each flow frame with CNN
        flow = flow.view(B * T, C, H, W)
        spatial_features = self.backbone(flow)  # (B*T, D, H', W')
        
        # Spatial pooling
        spatial_features = self.spatial_pool(spatial_features)  # (B*T, D, 1, 1)
        spatial_features = spatial_features.view(B * T, -1)  # (B*T, D)
        
        # Reshape for temporal modeling
        temporal_input = spatial_features.view(B, T, -1)  # (B, T, D)
        
        # Apply Bi-LSTM
        lstm_out, _ = self.temporal_lstm(temporal_input)  # (B, T, 512*2)
        
        # Apply attention to get weighted temporal features
        attention_weights = self.temporal_attention(lstm_out)  # (B, T, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum
        temporal_features = (lstm_out * attention_weights).sum(dim=1)  # (B, 512*2)
        
        # Classification
        logits = self.classifier(temporal_features)
        
        return logits, temporal_features


# Test the model
if __name__ == '__main__':
    model = OpticalFlowCNN(num_classes=6, num_frames=15)
    
    # Test input
    flow = torch.randn(2, 15, 2, 224, 224)  # Batch=2, Frames=15, Channels=2
    
    logits, features = model(flow)
    print(f"Flow shape: {flow.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")