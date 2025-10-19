import cv2
import os
import numpy as np
from tqdm import tqdm

def is_static_video(cap, num_samples=10, motion_threshold=2.0):
    """
    Checks if a video is mostly static (low motion).
    num_samples: number of frames sampled across the video.
    motion_threshold: average pixel difference below which video is considered static.
    """
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < num_samples:
        return True  # too short
    
    indices = np.linspace(0, frame_count - 1, num_samples).astype(int)
    prev_gray = None
    diffs = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            diffs.append(np.mean(diff))
        prev_gray = gray

    avg_diff = np.mean(diffs) if diffs else 0
    return avg_diff < motion_threshold


def filter_videos(input_root, output_root, min_duration=2, max_duration=30, min_res=224):
    """
    Filters and saves only good-quality videos into output_root, preserving folder structure.
    """
    for subdir, dirs, files in os.walk(input_root):
        for file in files:
            if not file.lower().endswith(".mp4"):
                continue

            input_path = os.path.join(subdir, file)
            rel_path = os.path.relpath(subdir, input_root)
            output_dir = os.path.join(output_root, rel_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file)

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"[WARN] Skipping unreadable: {input_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps == 0 or frame_count == 0:
                print(f"[WARN] Invalid FPS/frame count: {input_path}")
                continue

            duration = frame_count / fps

            # Filter conditions
            if duration < min_duration or duration > max_duration:
                print(f"[SKIP] Duration {duration:.2f}s out of range: {input_path}")
                continue

            if width < min_res or height < min_res:
                print(f"[SKIP] Low resolution ({width}x{height}): {input_path}")
                continue

            # Check for static video (low motion)
            if is_static_video(cap):
                print(f"[SKIP] Low motion detected: {input_path}")
                continue

            # Rewind video after motion check
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Prepare video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (224, 224))

            print(f"[INFO] Filtering {input_path} | Duration: {duration:.2f}s")

            for _ in tqdm(range(int(frame_count)), desc=f"Filtering {file}", unit="frame", leave=False):
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize to standard size (224x224)
                frame = cv2.resize(frame, (224, 224))

                # Optional denoising
                frame = cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)

                out.write(frame)

            cap.release()
            out.release()
            print(f"[SAVED] Filtered video to: {output_path}\n")

if __name__ == "__main__":
    input_root = "data/raw_videos"
    output_root = "data/filtered_videos"
    filter_videos(input_root, output_root)

