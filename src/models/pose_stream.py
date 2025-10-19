import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseGraphConv(nn.Module):
    """
    Graph Convolutional Network for skeleton pose
    Processes pose keypoints as graph nodes
    """
    def __init__(self, num_classes=6, num_frames=16, num_joints=33):
        super().__init__()
        self.num_joints = num_joints
        self.num_frames = num_frames
        
        # Build skeleton graph structure
        self.edge_index = self.build_skeleton_graph()
        
        # Graph convolution layers (process spatial structure)
        self.gcn1 = GraphConvLayer(4, 64)  # Input: x,y,z,visibility
        self.gcn2 = GraphConvLayer(64, 128)
        self.gcn3 = GraphConvLayer(128, 256)
        
        # Temporal convolution (process motion over time)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(256 * num_joints, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        
        # Joint angle processing branch
        self.angle_encoder = nn.Sequential(
            nn.Linear(10, 64),  # 10 key joint angles
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def build_skeleton_graph(self):
        """
        Build adjacency list for MediaPipe 33-keypoint skeleton
        Returns edge indices for graph convolution
        """
        # MediaPipe pose landmark connections
        # Reference: https://google.github.io/mediapipe/solutions/pose.html
        edges = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),  # Right eye
            (0, 4), (4, 5), (5, 6), (6, 8),  # Left eye
            (9, 10),  # Mouth
            
            # Torso
            (11, 12),  # Shoulders
            (11, 23), (12, 24),  # Shoulder to hip
            (23, 24),  # Hips
            
            # Right arm
            (12, 14), (14, 16),  # Shoulder-elbow-wrist
            (16, 18), (16, 20), (16, 22),  # Wrist to fingers
            (18, 20),  # Thumb
            
            # Left arm
            (11, 13), (13, 15),  # Shoulder-elbow-wrist
            (15, 17), (15, 19), (15, 21),  # Wrist to fingers
            (17, 19),  # Thumb
            
            # Right leg
            (24, 26), (26, 28),  # Hip-knee-ankle
            (28, 30), (28, 32),  # Ankle to foot
            (30, 32),  # Foot
            
            # Left leg
            (23, 25), (25, 27),  # Hip-knee-ankle
            (27, 29), (27, 31),  # Ankle to foot
            (29, 31),  # Foot
        ]
        
        # Make edges bidirectional
        edges_bidirectional = edges + [(b, a) for a, b in edges]
        
        edge_index = torch.tensor(edges_bidirectional, dtype=torch.long).t()
        return edge_index
    
    def forward(self, pose_sequence, joint_angles):
        """
        Args:
            pose_sequence: (B, T, J, 4) - keypoints with x,y,z,visibility
            joint_angles: (B, T, A) - computed joint angles
        
        Returns:
            logits: (B, num_classes)
            features: (B, 640) - for fusion
        """
        B, T, J, F = pose_sequence.shape
        
        # Process each frame with spatial GCN
        spatial_features_list = []
        
        for t in range(T):
            # Get frame t for all batches
            frame_poses = pose_sequence[:, t, :, :]  # (B, J, 4)
            
            # Flatten for GCN processing
            frame_poses_flat = frame_poses.reshape(B * J, F)  # (B*J, 4)
            
            # Apply GCN layers
            edge_index = self.edge_index.to(frame_poses_flat.device)
            
            x = F.relu(self.gcn1(frame_poses_flat, edge_index, B))
            x = F.relu(self.gcn2(x, edge_index, B))
            x = F.relu(self.gcn3(x, edge_index, B))  # (B*J, 256)
            
            # Reshape back to (B, J, 256)
            x = x.view(B, J, -1)
            spatial_features_list.append(x)
        
        # Stack temporal dimension: (B, T, J, 256)
        spatial_features = torch.stack(spatial_features_list, dim=1)
        
        # Flatten joints for temporal processing: (B, T, J*256)
        spatial_features = spatial_features.view(B, T, -1)
        
        # Permute for Conv1d: (B, J*256, T)
        spatial_features = spatial_features.permute(0, 2, 1)
        
        # Apply temporal convolution
        temporal_features = self.temporal_conv(spatial_features)  # (B, 512, T)
        
        # Global average pooling over time
        temporal_features = temporal_features.mean(dim=2)  # (B, 512)
        
        # Process joint angles
        # Average angles over time
        angle_features = joint_angles.mean(dim=1)  # (B, A)
        angle_features = self.angle_encoder(angle_features)  # (B, 128)
        
        # Concatenate pose and angle features
        combined_features = torch.cat([temporal_features, angle_features], dim=1)  # (B, 640)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits, combined_features


class GraphConvLayer(nn.Module):
    """Simple Graph Convolution Layer"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, x, edge_index, batch_size):
        """
        Args:
            x: (N, in_features) node features
            edge_index: (2, E) edge connections
            batch_size: batch size for proper aggregation
        """
        # Linear transformation
        x = self.linear(x)
        
        # Message passing: aggregate neighbor features
        row, col = edge_index
        
        # Simple mean aggregation
        num_nodes = x.size(0)
        aggregated = torch.zeros_like(x)
        
        # Count neighbors for averaging
        neighbor_count = torch.zeros(num_nodes, 1, device=x.device)
        
        # Sum neighbors
        aggregated.index_add_(0, row, x[col])
        neighbor_count.index_add_(0, row, torch.ones((edge_index.size(1), 1), device=x.device))
        
        # Average
        aggregated = aggregated / (neighbor_count + 1e-8)
        
        # Add self-connection
        x = x + aggregated
        
        # Batch normalization
        x = self.bn(x)
        
        return x


# Test the model
if __name__ == '__main__':
    model = PoseGraphConv(num_classes=6, num_frames=16)
    
    # Test inputs
    pose_seq = torch.randn(2, 16, 33, 4)  # Batch=2, Frames=16, Joints=33
    angles = torch.randn(2, 16, 10)  # Batch=2, Frames=16, Angles=10
    
    logits, features = model(pose_seq, angles)
    print(f"Pose shape: {pose_seq.shape}")
    print(f"Angles shape: {angles.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
