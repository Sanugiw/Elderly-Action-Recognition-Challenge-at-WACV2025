import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from decord import VideoReader, cpu
import cv2

class MultiModalDataset(Dataset):
    """
    Dataset that loads RGB, Pose, and Optical Flow together
    Ensures synchronized sampling across all modalities
    """
    def __init__(
        self,
        video_dir,
        pose_dir,
        flow_dir,
        video_ids=None,
        labels=None,
        transform=None,
        num_frames=16
    ):
        """
        Args:
            video_dir: Directory containing RGB videos
            pose_dir: Directory containing pose .npy files
            flow_dir: Directory containing flow .npy files
            video_ids: List of video IDs to load (None = all)
            labels: Dictionary mapping video_id to label
            transform: Optional transforms
            num_frames: Number of frames to sample
        """
        self.video_dir = Path(video_dir)
        self.pose_dir = Path(pose_dir)
        self.flow_dir = Path(flow_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.labels = labels
        
        # Get video list
        if video_ids is None:
            video_paths = list(self.video_dir.glob('*.mp4'))
            self.video_ids = [v.stem for v in video_paths]
        else:
            self.video_ids = video_ids
        
        print(f"MultiModalDataset initialized with {len(self.video_ids)} samples")
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # Load RGB video
        rgb_frames = self._load_rgb(video_id)
        
        # Load pose data
        pose_sequence, joint_angles = self._load_pose(video_id)
        
        # Load optical flow
        flow_sequence = self._load_flow(video_id)
        
        # Apply transforms if provided
        if self.transform:
            rgb_frames, pose_sequence, flow_sequence = self.transform(
                rgb_frames, pose_sequence, flow_sequence
            )
        
        # Convert to tensors
        rgb_tensor = self._to_tensor_rgb(rgb_frames)
        pose_tensor = torch.from_numpy(pose_sequence).float()
        angles_tensor = torch.from_numpy(joint_angles).float()
        flow_tensor = torch.from_numpy(flow_sequence).float()
        
        # Permute flow: (T, H, W, 2) -> (T, 2, H, W)
        flow_tensor = flow_tensor.permute(0, 3, 1, 2)
        
        # Get label if available
        if self.labels is not None:
            label = self.labels.get(video_id, 0)  # Default to 0 if missing
            return rgb_tensor, pose_tensor, angles_tensor, flow_tensor, label, video_id
        
        return rgb_tensor, pose_tensor, angles_tensor, flow_tensor, video_id
    
    def _load_rgb(self, video_id):
        """Load RGB video frames"""
        video_path = self.video_dir / f"{video_id}.mp4"
        
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)
            
            # Sample uniformly
            if total_frames < self.num_frames:
                indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
            else:
                indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
            
            frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
            
            # Resize to 224x224
            resized_frames = []
            for frame in frames:
                resized = cv2.resize(frame, (224, 224))
                resized_frames.append(resized)
            
            return np.stack(resized_frames)
            
        except Exception as e:
            print(f"Error loading RGB for {video_id}: {e}")
            return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
    
    def _load_pose(self, video_id):
        """Load pose keypoints and joint angles"""
        pose_path = self.pose_dir / f"{video_id}_pose.npy"
        angle_path = self.pose_dir / f"{video_id}_angles.npy"
        
        try:
            pose_sequence = np.load(pose_path)  # (T, 33, 4)
            joint_angles = np.load(angle_path)  # (T, 10)
            return pose_sequence, joint_angles
        except Exception as e:
            print(f"Error loading pose for {video_id}: {e}")
            return (
                np.zeros((self.num_frames, 33, 4), dtype=np.float32),
                np.zeros((self.num_frames, 10), dtype=np.float32)
            )
    
    def _load_flow(self, video_id):
        """Load optical flow"""
        flow_path = self.flow_dir / f"{video_id}_flow.npy"
        
        try:
            flow_sequence = np.load(flow_path)  # (T-1, H, W, 2)
            
            # Ensure correct size
            if flow_sequence.shape[0] < self.num_frames - 1:
                # Pad with zeros
                padding = np.zeros(
                    (self.num_frames - 1 - flow_sequence.shape[0], 224, 224, 2),
                    dtype=np.float32
                )
                flow_sequence = np.concatenate([flow_sequence, padding], axis=0)
            elif flow_sequence.shape[0] > self.num_frames - 1:
                # Truncate
                flow_sequence = flow_sequence[:self.num_frames - 1]
            
            return flow_sequence
            
        except Exception as e:
            print(f"Error loading flow for {video_id}: {e}")
            return np.zeros((self.num_frames - 1, 224, 224, 2), dtype=np.float32)
    
    def _to_tensor_rgb(self, frames):
        """Convert RGB frames to tensor"""
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # (T, H, W, C) -> (T, C, H, W)
        frames = np.transpose(frames, (0, 3, 1, 2))
        
        return torch.from_numpy(frames).float()


def create_dataloaders(
    video_dir='data/videos',
    pose_dir='data/poses',
    flow_dir='data/optical_flow',
    label_file=None,
    batch_size=4,
    num_workers=4,
    val_split=0.1,
    num_frames=16
):
    """
    Create train and validation dataloaders
    
    Args:
        video_dir: Path to video directory
        pose_dir: Path to pose directory
        flow_dir: Path to flow directory
        label_file: Path to CSV with video_id,label columns
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_split: Validation split ratio
        num_frames: Number of frames per video
    
    Returns:
        train_loader, val_loader
    """
    import pandas as pd
    
    # Load labels if available (for training data)
    # For now, we'll use test data with dummy labels
    video_dir = Path(video_dir)
    video_paths = list(video_dir.glob('*.mp4'))
    video_ids = [v.stem for v in video_paths]
    
    # Create dummy labels for testing (replace with real labels)
    labels = {vid: idx % 6 for idx, vid in enumerate(video_ids)}
    
    print(f"Found {len(video_ids)} videos")
    
    # Split train/val
    from sklearn.model_selection import train_test_split
    train_ids, val_ids = train_test_split(
        video_ids,
        test_size=val_split,
        random_state=42
    )
    
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    
    # Create datasets
    train_dataset = MultiModalDataset(
        video_dir=video_dir,
        pose_dir=pose_dir,
        flow_dir=flow_dir,
        video_ids=train_ids,
        labels=labels,
        num_frames=num_frames
    )
    
    val_dataset = MultiModalDataset(
        video_dir=video_dir,
        pose_dir=pose_dir,
        flow_dir=flow_dir,
        video_ids=val_ids,
        labels=labels,
        num_frames=num_frames
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # For stable batch norm
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# Test the dataset
if __name__ == '__main__':
    train_loader, val_loader = create_dataloaders(
        batch_size=2,
        num_workers=2
    )
    
    print("Testing dataset loading...")
    for batch in train_loader:
        rgb, pose, angles, flow, labels, video_ids = batch
        print(f"RGB shape: {rgb.shape}")
        print(f"Pose shape: {pose.shape}")
        print(f"Angles shape: {angles.shape}")
        print(f"Flow shape: {flow.shape}")
        print(f"Labels: {labels}")
        print(f"Video IDs: {video_ids}")
        break
    
    print("✓ Dataset loading successful!")
