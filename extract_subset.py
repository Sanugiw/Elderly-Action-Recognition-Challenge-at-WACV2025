import os
import random
import shutil
from tqdm import tqdm

def extract_subset(input_root, output_root, fraction=0.25, extensions=(".mp4", ".avi", ".mov")):
    """
    Copies a random fraction of video files from input_root to output_root,
    preserving the directory structure.
    """
    all_videos = []

    # Traverse folders and collect video paths
    for root, dirs, files in os.walk(input_root):
        for f in files:
            if f.lower().endswith(extensions):
                all_videos.append(os.path.join(root, f))

    print(f"Found {len(all_videos)} videos in total.")

    # Shuffle and select subset
    random.shuffle(all_videos)
    subset_count = int(len(all_videos) * fraction)
    subset_videos = all_videos[:subset_count]

    print(f"Extracting {subset_count} videos ({fraction*100:.0f}%)...")

    for src_path in tqdm(subset_videos, desc="Copying videos"):
        # Recreate folder structure in output
        rel_path = os.path.relpath(src_path, input_root)
        dest_path = os.path.join(output_root, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src_path, dest_path)

    print(f"✅ Extraction complete! Subset saved to: {output_root}")

if __name__ == "__main__":
    extract_subset(
        input_root="data/raw_videos",         # path to your original dataset
        output_root="data/subset_videos",     # where subset will be saved
        fraction=0.1                         # 1/4 of the dataset
    )
