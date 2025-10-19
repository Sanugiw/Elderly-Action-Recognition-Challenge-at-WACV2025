import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

class OpticalFlowExtractor:
    """Extract optical flow from videos using Farneback method"""
    def __init__(self, method='farneback'):
        """
        Args:
            method: 'farneback' or 'lucaskanade'
        """
        self.method = method
    
    def extract_flow_sequence(self, video_path, num_frames=16):
        """
        Extract optical flow between consecutive frames
        
        Returns:
            flow: numpy array of shape (num_frames-1, H, W, 2)
                  2 channels: flow_x, flow_y
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print(f"Warning: Could not read {video_path}")
            return np.zeros((num_frames-1, 224, 224, 2))
        
        # Sample frame indices
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        flows = []
        prev_gray = None
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (224, 224))
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Calculate optical flow
                if self.method == 'farneback':
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5,      # Image scale for pyramid
                        levels=3,           # Number of pyramid layers
                        winsize=15,         # Window size
                        iterations=3,       # Iterations at each pyramid level
                        poly_n=5,           # Polynomial expansion
                        poly_sigma=1.2,     # Gaussian SD for polynomial expansion
                        flags=0
                    )
                    flows.append(flow)
                
            prev_gray = gray
        
        cap.release()
        
        if len(flows) == 0:
            return np.zeros((num_frames-1, 224, 224, 2))
        
        # Pad if necessary
        while len(flows) < num_frames - 1:
            flows.append(np.zeros((224, 224, 2)))
        
        return np.array(flows)
    
    def flow_to_rgb(self, flow):
        """
        Convert optical flow to RGB image for visualization
        
        Args:
            flow: (H, W, 2) flow array
        
        Returns:
            rgb: (H, W, 3) RGB visualization
        """
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255  # Full saturation
        
        # Calculate magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Map angle to hue
        hsv[..., 0] = ang * 180 / np.pi / 2
        
        # Map magnitude to value
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return rgb
    
    def normalize_flow(self, flow):
        """
        Normalize flow values to [-1, 1] range
        
        Args:
            flow: (T, H, W, 2) flow array
        
        Returns:
            normalized_flow: (T, H, W, 2) normalized flow
        """
        # Clip extreme values
        flow = np.clip(flow, -20, 20)
        
        # Normalize to [-1, 1]
        flow = flow / 20.0
        
        return flow

def process_all_videos_flow(video_dir, output_dir, num_frames=16):
    """
    Extract optical flow from all videos
    
    Args:
        video_dir: Directory containing videos
        output_dir: Directory to save flow data
        num_frames: Number of frames to sample per video
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all videos
    video_extensions = ['.mp4', '.avi', '.mov']
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(list(video_dir.glob(f'*{ext}')))
    
    print(f"Found {len(video_paths)} videos to process")
    
    # Initialize extractor
    extractor = OpticalFlowExtractor(method='farneback')
    
    # Process each video
    for video_path in tqdm(video_paths, desc="Extracting optical flow"):
        video_id = video_path.stem
        
        # Check if already processed
        flow_file = output_dir / f"{video_id}_flow.npy"
        
        if flow_file.exists():
            continue  # Skip if already processed
        
        try:
            # Extract flow sequence
            flows = extractor.extract_flow_sequence(video_path, num_frames)
            
            # Normalize flow
            flows = extractor.normalize_flow(flows)
            
            # Save
            np.save(flow_file, flows)
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            # Save empty array on error
            np.save(flow_file, np.zeros((num_frames-1, 224, 224, 2)))
    
    print(f"Optical flow extraction completed! Saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract optical flow from videos')
    parser.add_argument('--video_dir', type=str, default='data/videos',
                       help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='data/optical_flow',
                       help='Directory to save flow data')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames to sample per video')
    
    args = parser.parse_args()
    
    process_all_videos_flow(args.video_dir, args.output_dir, args.num_frames)
