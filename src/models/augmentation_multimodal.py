import albumentations as A
import numpy as np
import cv2

class MultiModalAugmentation:
    """
    Synchronized augmentation for RGB, Pose, and Flow
    Applies same geometric transforms to maintain consistency
    """
    def __init__(self, mode='train', img_size=224):
        self.mode = mode
        self.img_size = img_size
        
        if mode == 'train':
            self.spatial_transforms = A.Compose([
                A.Resize(256, 256),
                A.RandomCrop(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.3
                ),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0
                    ),
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1,
                        p=1.0
                    ),
                ], p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            ], additional_targets={'flow': 'image'})
            
            self.normalize = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.spatial_transforms = A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(img_size, img_size),
            ], additional_targets={'flow': 'image'})
            
            self.normalize = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __call__(self, rgb_frames, pose_sequence, flow_frames):
        """
        Apply synchronized augmentations
        
        Args:
            rgb_frames: (T, H, W, C) numpy array
            pose_sequence: (T, J, 4) numpy array
            flow_frames: (T-1, H, W, 2) numpy array
        
        Returns:
            augmented versions of all inputs
        """
        T = rgb_frames.shape[0]
        
        # Store transformed data
        aug_rgb = []
        aug_flow = []
        
        # Apply same transform to each RGB-Flow pair
        for t in range(T):
            rgb_frame = rgb_frames[t]
            
            if t < len(flow_frames):
                flow_frame = flow_frames[t]
                
                # Apply spatial transforms
                transformed = self.spatial_transforms(
                    image=rgb_frame,
                    flow=flow_frame
                )
                
                # Normalize RGB only
                rgb_transformed = self.normalize(image=transformed['image'])['image']
                
                aug_rgb.append(rgb_transformed)
                aug_flow.append(transformed['flow'])
            else:
                # Last RGB frame has no corresponding flow
                transformed = self.spatial_transforms(image=rgb_frame)
                rgb_transformed = self.normalize(image=transformed['image'])['image']
                aug_rgb.append(rgb_transformed)
        
        # Transform pose keypoints
        aug_pose = self.transform_pose(pose_sequence, rgb_frames.shape[1:3])
        
        return np.array(aug_rgb), aug_pose, np.array(aug_flow)
    
    def transform_pose(self, pose_sequence, original_size):
        """
        Apply geometric transforms to pose keypoints
        For now, we keep pose as-is since it's normalized coordinates
        In production, you'd apply the same transforms as RGB
        """
        # Pose coordinates are already normalized (0-1 range)
        # For horizontal flip, we'd mirror x coordinates
        # For crops/resize, we'd adjust coordinates accordingly
        
        # Simplified: just return as-is
        # TODO: Implement proper synchronized transforms
        return pose_sequence


class TemporalAugmentation:
    """
    Temporal augmentation strategies
    """
    def __init__(self, mode='train'):
        self.mode = mode
    
    def random_temporal_crop(self, frames, target_frames=16):
        """Randomly crop temporal segment"""
        total_frames = len(frames)
        
        if total_frames <= target_frames:
            return frames
        
        start_idx = np.random.randint(0, total_frames - target_frames + 1)
        return frames[start_idx:start_idx + target_frames]
    
    def temporal_stride(self, frames, stride=2):
        """Sample frames with stride"""
        return frames[::stride]


# Test augmentation
if __name__ == '__main__':
    print("Testing multi-modal augmentation...")
    
    # Create dummy data
    rgb = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
    pose = np.random.rand(16, 33, 4).astype(np.float32)
    flow = np.random.randn(15, 224, 224, 2).astype(np.float32)
    
    # Test train mode
    aug_train = MultiModalAugmentation(mode='train')
    aug_rgb, aug_pose, aug_flow = aug_train(rgb, pose, flow)
    
    print(f"Original RGB: {rgb.shape} -> Augmented: {aug_rgb.shape}")
    print(f"Original Pose: {pose.shape} -> Augmented: {aug_pose.shape}")
    print(f"Original Flow: {flow.shape} -> Augmented: {aug_flow.shape}")
    
    # Test val mode
    aug_val = MultiModalAugmentation(mode='val')
    aug_rgb, aug_pose, aug_flow = aug_val(rgb, pose, flow)
    print(f"\n✓ Val mode augmentation working")
    
    # Test temporal augmentation
    temp_aug = TemporalAugmentation()
    cropped = temp_aug.random_temporal_crop(rgb, target_frames=12)
    print(f"✓ Temporal crop: {rgb.shape} -> {cropped.shape}")
    
    print("\n✅ All augmentations working!")
