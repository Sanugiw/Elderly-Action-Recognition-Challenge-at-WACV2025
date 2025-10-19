import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class OpticalFlowDataset(Dataset):
    """Dataset for loading optical flow"""
    def __init__(self, flow_dir, video_ids, labels=None, transform=None):
        """
        Args:
            flow_dir: Directory containing flow .npy files
            video_ids: List of video IDs
            labels: List of labels (None for test set)
            transform: Optional transforms
        """
        self.flow_dir = Path(flow_dir)
        self.video_ids = video_ids
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # Load flow data
        flow_path = self.flow_dir / f"{video_id}_flow.npy"
        
        try:
            flow = np.load(flow_path)  # (T, H, W, 2)
            
            # Apply transforms if provided
            if self.transform:
                # Apply same transform to all flow frames
                pass  # Implement if needed
            
            # Convert to tensor: (T, H, W, 2) -> (T, 2, H, W)
            flow = torch.from_numpy(flow).permute(0, 3, 1, 2).float()
            
        except Exception as e:
            print(f"Error loading {flow_path}: {e}")
            # Return dummy data if flow fails to load
            flow = torch.zeros(15, 2, 224, 224)  # 15 frames (16-1)
        
        if self.labels is not None:
            return flow, self.labels[idx], video_id
        return flow, video_id

# Test the loader
if __name__ == '__main__':
    from pathlib import Path
    
    flow_dir = Path('data/optical_flow')
    flow_files = list(flow_dir.glob('*_flow.npy'))
    video_ids = [f.stem.replace('_flow', '') for f in flow_files[:5]]
    
    print(f"Found {len(flow_files)} flow files")
    
    dataset = OpticalFlowDataset(flow_dir, video_ids)
    
    for i in range(min(2, len(dataset))):
        flow, video_id = dataset[i]
        print(f"Video {i}: {video_id}, Shape: {flow.shape}")
