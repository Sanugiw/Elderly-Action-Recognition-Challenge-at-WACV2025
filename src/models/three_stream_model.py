import torch
import torch.nn as nn

from models.rgb_stream import RGBTransformer
from models.pose_stream import PoseGraphConv
from models.flow_stream import OpticalFlowCNN
from models.fusion_module import AttentionFusion

class ThreeStreamModel(nn.Module):
    """
    Complete three-stream model with fusion
    Combines RGB, Pose, and Optical Flow streams
    """
    def __init__(
        self,
        num_classes=6,
        num_frames=16,
        fusion_type='attention',
        pretrained=True
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.fusion_type = fusion_type
        
        # Initialize three streams
        self.rgb_stream = RGBTransformer(
            num_classes=num_classes,
            num_frames=num_frames,
            pretrained=pretrained
        )
        
        self.pose_stream = PoseGraphConv(
            num_classes=num_classes,
            num_frames=num_frames
        )
        
        self.flow_stream = OpticalFlowCNN(
            num_classes=num_classes,
            num_frames=num_frames - 1  # Flow has T-1 frames
        )
        
        # Fusion module
        if fusion_type == 'attention':
            self.fusion = AttentionFusion(
                rgb_dim=768,
                pose_dim=640,
                flow_dim=1024,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, rgb, pose, angles, flow, return_stream_outputs=False):
        """
        Forward pass through all streams
        
        Args:
            rgb: (B, T, C, H, W)
            pose: (B, T, J, 4)
            angles: (B, T, A)
            flow: (B, T-1, 2, H, W)
            return_stream_outputs: If True, return individual stream predictions
        
        Returns:
            logits: (B, num_classes)
            (optional) stream_outputs: dict with individual stream results
        """
        # Extract features from each stream
        rgb_logits, rgb_features = self.rgb_stream(rgb)
        pose_logits, pose_features = self.pose_stream(pose, angles)
        flow_logits, flow_features = self.flow_stream(flow)
        
        # Fuse features
        if self.fusion_type == 'attention':
            fused_logits, gate_weights = self.fusion(
                rgb_features,
                pose_features,
                flow_features
            )
        
        if return_stream_outputs:
            stream_outputs = {
                'rgb': rgb_logits,
                'pose': pose_logits,
                'flow': flow_logits,
                'gate_weights': gate_weights if self.fusion_type == 'attention' else None
            }
            return fused_logits, stream_outputs
        
        return fused_logits
    
    def freeze_stream(self, stream_name):
        """Freeze parameters of a specific stream"""
        if stream_name == 'rgb':
            for param in self.rgb_stream.parameters():
                param.requires_grad = False
        elif stream_name == 'pose':
            for param in self.pose_stream.parameters():
                param.requires_grad = False
        elif stream_name == 'flow':
            for param in self.flow_stream.parameters():
                param.requires_grad = False
    
    def unfreeze_stream(self, stream_name):
        """Unfreeze parameters of a specific stream"""
        if stream_name == 'rgb':
            for param in self.rgb_stream.parameters():
                param.requires_grad = True
        elif stream_name == 'pose':
            for param in self.pose_stream.parameters():
                param.requires_grad = True
        elif stream_name == 'flow':
            for param in self.flow_stream.parameters():
                param.requires_grad = True
    
    def load_stream_checkpoint(self, stream_name, checkpoint_path):
        """Load pretrained weights for individual stream"""
        checkpoint = torch.load(checkpoint_path)
        
        if stream_name == 'rgb':
            self.rgb_stream.load_state_dict(checkpoint['model_state_dict'])
        elif stream_name == 'pose':
            self.pose_stream.load_state_dict(checkpoint['model_state_dict'])
        elif stream_name == 'flow':
            self.flow_stream.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded {stream_name} stream from {checkpoint_path}")


# Test the three-stream model
if __name__ == '__main__':
    model = ThreeStreamModel(num_classes=6)
    
    # Test inputs
    rgb = torch.randn(2, 16, 3, 224, 224)
    pose = torch.randn(2, 16, 33, 4)
    angles = torch.randn(2, 16, 10)
    flow = torch.randn(2, 15, 2, 224, 224)
    
    # Forward pass
    logits = model(rgb, pose, angles, flow)
    print(f"Input shapes:")
    print(f"  RGB: {rgb.shape}")
    print(f"  Pose: {pose.shape}")
    print(f"  Angles: {angles.shape}")
    print(f"  Flow: {flow.shape}")
    print(f"\nOutput logits: {logits.shape}")
    
    # Test with stream outputs
    logits, stream_outputs = model(rgb, pose, angles, flow, return_stream_outputs=True)
    print(f"\nStream outputs:")
    for name, output in stream_outputs.items():
        if output is not None:
            print(f"  {name}: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    print("\n✅ Three-stream model working!")
