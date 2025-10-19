import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import numpy as np
from pathlib import Path
import cv2

class RGBDataset(Dataset):
    """Dataset for loading RGB video frames"""
    def __init__(self, video_paths, labels=None, transform=None, num_frames=16):
        """
        Args:
            video_paths: List of video file paths
            labels: List of labels (None for test set)
            transform: Augmentation transforms
            num_frames: Number of frames to sample per video
        """
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = str(self.video_paths[idx])
        video_id = Path(video_path).stem
        
        try:
            # Load video with decord (much faster than OpenCV)
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            # Sample frames uniformly across the video
            if total_frames < self.num_frames:
                # If video is shorter, repeat frames
                indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
            else:
                indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
            
            # Get frames
            frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
            
            # Apply transforms if provided
            if self.transform:
                transformed_frames = []
                for frame in frames:
                    transformed = self.transform(image=frame)['image']
                    transformed_frames.append(transformed)
                frames = np.stack(transformed_frames)
            else:
                # Default: normalize to [0, 1] and resize
                frames = [cv2.resize(f, (224, 224)) for f in frames]
                frames = np.stack(frames).astype(np.float32) / 255.0
            
            # Convert to tensor: (T, H, W, C) -> (T, C, H, W)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # Return dummy data if video fails to load
            frames = torch.zeros(self.num_frames, 3, 224, 224)
        
        if self.labels is not None:
            return frames, self.labels[idx], video_id
        return frames, video_id
    
def get_video_paths(data_dir):
    """Get all video paths from directory"""
    data_dir = Path(data_dir)
    video_extensions = ['.mp4', '.avi', '.mov']
    
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(list(data_dir.glob(f'*{ext}')))
    
    return sorted(video_paths)

# Test the loader
if __name__ == '__main__':
    video_paths = get_video_paths('data/videos')
    print(f"Found {len(video_paths)} videos")
    
    dataset = RGBDataset(video_paths[:5])  # Test with 5 videos
    
    for i in range(min(2, len(dataset))):
        frames, video_id = dataset[i]
        print(f"Video {i}: {video_id}, Shape: {frames.shape}")