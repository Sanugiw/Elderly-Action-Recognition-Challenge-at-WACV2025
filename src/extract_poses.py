import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse

class PoseExtractor:
    """Extract pose keypoints using MediaPipe"""
    def __init__(self, model_complexity=2):
        """
        Args:
            model_complexity: 0, 1, or 2. Higher = more accurate but slower
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True
        )
    
    def extract_pose_sequence(self, video_path, num_frames=16):
        """
        Extract pose keypoints from video
        
        Returns:
            poses: numpy array of shape (num_frames, 33, 4)
                   33 keypoints, each with (x, y, z, visibility)
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print(f"Warning: Could not read {video_path}")
            return np.zeros((num_frames, 33, 4))
        
        # Sample frame indices uniformly
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        poses = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                # If frame read fails, use zeros
                poses.append(np.zeros((33, 4)))
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Extract 33 keypoints
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
                poses.append(np.array(landmarks))
            else:
                # No pose detected, use zeros
                poses.append(np.zeros((33, 4)))
        
        cap.release()
        return np.array(poses)
    
    def compute_joint_angles(self, pose_sequence):
        """
        Compute key joint angles from pose keypoints
        
        Returns:
            angles: numpy array of shape (num_frames, 10)
                    10 key angles for elderly activity recognition
        """
        num_frames = pose_sequence.shape[0]
        angles = []
        
        for frame_idx in range(num_frames):
            kpts = pose_sequence[frame_idx]  # (33, 4)
            
            frame_angles = []
            
            # Left elbow angle (shoulder-elbow-wrist)
            if kpts[11, 3] > 0.5 and kpts[13, 3] > 0.5 and kpts[15, 3] > 0.5:
                angle = self.calculate_angle(
                    kpts[11, :2], kpts[13, :2], kpts[15, :2]
                )
                frame_angles.append(angle)
            else:
                frame_angles.append(0.0)
            
            # Right elbow angle
            if kpts[12, 3] > 0.5 and kpts[14, 3] > 0.5 and kpts[16, 3] > 0.5:
                angle = self.calculate_angle(
                    kpts[12, :2], kpts[14, :2], kpts[16, :2]
                )
                frame_angles.append(angle)
            else:
                frame_angles.append(0.0)
            
            # Left knee angle (hip-knee-ankle)
            if kpts[23, 3] > 0.5 and kpts[25, 3] > 0.5 and kpts[27, 3] > 0.5:
                angle = self.calculate_angle(
                    kpts[23, :2], kpts[25, :2], kpts[27, :2]
                )
                frame_angles.append(angle)
            else:
                frame_angles.append(0.0)
            
            # Right knee angle
            if kpts[24, 3] > 0.5 and kpts[26, 3] > 0.5 and kpts[28, 3] > 0.5:
                angle = self.calculate_angle(
                    kpts[24, :2], kpts[26, :2], kpts[28, :2]
                )
                frame_angles.append(angle)
            else:
                frame_angles.append(0.0)
            
            # Left shoulder angle (elbow-shoulder-hip)
            if kpts[13, 3] > 0.5 and kpts[11, 3] > 0.5 and kpts[23, 3] > 0.5:
                angle = self.calculate_angle(
                    kpts[13, :2], kpts[11, :2], kpts[23, :2]
                )
                frame_angles.append(angle)
            else:
                frame_angles.append(0.0)
            
            # Right shoulder angle
            if kpts[14, 3] > 0.5 and kpts[12, 3] > 0.5 and kpts[24, 3] > 0.5:
                angle = self.calculate_angle(
                    kpts[14, :2], kpts[12, :2], kpts[24, :2]
                )
                frame_angles.append(angle)
            else:
                frame_angles.append(0.0)
            
            # Spine angle (shoulder-hip-knee)
            if kpts[11, 3] > 0.5 and kpts[23, 3] > 0.5 and kpts[25, 3] > 0.5:
                angle = self.calculate_angle(
                    kpts[11, :2], kpts[23, :2], kpts[25, :2]
                )
                frame_angles.append(angle)
            else:
                frame_angles.append(0.0)
            
            # Left hip angle (shoulder-hip-knee)
            if kpts[11, 3] > 0.5 and kpts[23, 3] > 0.5 and kpts[25, 3] > 0.5:
                angle = self.calculate_angle(
                    kpts[11, :2], kpts[23, :2], kpts[25, :2]
                )
                frame_angles.append(angle)
            else:
                frame_angles.append(0.0)
            
            # Right hip angle
            if kpts[12, 3] > 0.5 and kpts[24, 3] > 0.5 and kpts[26, 3] > 0.5:
                angle = self.calculate_angle(
                    kpts[12, :2], kpts[24, :2], kpts[26, :2]
                )
                frame_angles.append(angle)
            else:
                frame_angles.append(0.0)
            
            # Neck angle (nose-shoulder midpoint)
            if kpts[0, 3] > 0.5 and kpts[11, 3] > 0.5 and kpts[12, 3] > 0.5:
                shoulder_mid = (kpts[11, :2] + kpts[12, :2]) / 2
                angle = self.calculate_angle(
                    kpts[0, :2], shoulder_mid, kpts[11, :2]
                )
                frame_angles.append(angle)
            else:
                frame_angles.append(0.0)
            
            angles.append(frame_angles)
        
        return np.array(angles)
    
    def calculate_angle(self, a, b, c):
        """
        Calculate angle at point b formed by points a-b-c
        
        Args:
            a, b, c: 2D points (x, y)
        
        Returns:
            angle in degrees
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def close(self):
        """Release resources"""
        self.pose.close()

def process_all_videos(video_dir, output_dir, num_frames=16):
    """
    Extract poses from all videos in directory
    
    Args:
        video_dir: Directory containing videos
        output_dir: Directory to save pose data
        num_frames: Number of frames to extract per video
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
    extractor = PoseExtractor(model_complexity=2)
    
    # Process each video
    for video_path in tqdm(video_paths, desc="Extracting poses"):
        video_id = video_path.stem
        
        # Check if already processed
        pose_file = output_dir / f"{video_id}_pose.npy"
        angle_file = output_dir / f"{video_id}_angles.npy"
        
        if pose_file.exists() and angle_file.exists():
            continue  # Skip if already processed
        
        try:
            # Extract pose sequence
            poses = extractor.extract_pose_sequence(video_path, num_frames)
            
            # Compute joint angles
            angles = extractor.compute_joint_angles(poses)
            
            # Save
            np.save(pose_file, poses)
            np.save(angle_file, angles)
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            # Save empty arrays on error
            np.save(pose_file, np.zeros((num_frames, 33, 4)))
            np.save(angle_file, np.zeros((num_frames, 10)))
    
    extractor.close()
    print(f"Pose extraction completed! Saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract poses from videos')
    parser.add_argument('--video_dir', type=str, default='data/videos',
                       help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='data/poses',
                       help='Directory to save pose data')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames to extract per video')
    
    args = parser.parse_args()
    
    process_all_videos(args.video_dir, args.output_dir, args.num_frames)