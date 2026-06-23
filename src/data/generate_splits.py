"""
generate_splits.py
==================
Generates train/val/test splits (70/10/20) for the SSL400 dataset.
Pure Python implementation (no pandas/sklearn required).
"""

import csv
import random
from pathlib import Path

# Fix seed for reproducible splits
random.seed(42)

def main():
    base_dir = Path("SSL400/Dataset - Original")
    output_dir = Path("data/splits")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect classes from folders
    all_classes = []
    for category_dir in base_dir.iterdir():
        if category_dir.is_dir():
            for word_dir in category_dir.iterdir():
                if word_dir.is_dir():
                    all_classes.append(word_dir.name)
    all_classes = sorted(list(set(all_classes)))
    
    # Auto-generate or load Sinhala word map
    class_map = {}
    map_path = output_dir / "sinhala_word_map.csv"
    
    if map_path.exists():
        with open(map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_map[row["class_name_english"]] = int(row["class_id"])
                
        # If the map doesn't have our current folders, we need to regenerate
        if not all(c in class_map for c in all_classes):
            print("Existing word map does not match current folders. Regenerating...")
            class_map = {}
            
    if not class_map:
        print(f"Generating new word map for {len(all_classes)} classes...")
        with open(map_path, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name_sinhala", "class_name_english"])
            for idx, c in enumerate(all_classes):
                writer.writerow([idx, c, c]) # Placeholder sinhala name
                class_map[c] = idx

    # Gather all video files
    all_videos = []
    for ext in ["*.mp4", "*.avi", "*.mov"]:
        for file in base_dir.rglob(ext):
            class_name = file.parent.name
            if class_name in class_map:
                all_videos.append({
                    "video_path": str(file).replace("\\", "/"),
                    "class_name": class_name,
                    "class_id": class_map[class_name]
                })

    print(f"Found {len(all_videos)} valid videos across {len(class_map)} classes.")

    # Group by class
    by_class = {}
    for v in all_videos:
        c_id = v["class_id"]
        if c_id not in by_class:
            by_class[c_id] = []
        by_class[c_id].append(v)

    # Stratified split
    train_data, val_data, test_data = [], [], []

    for c_id, vids in by_class.items():
        random.shuffle(vids)
        total = len(vids)
        
        train_count = int(total * 0.70)
        val_count = int(total * 0.10)
        
        # Ensure at least 1 for val and test if possible
        if val_count == 0 and total >= 3:
            val_count = 1
        if (total - train_count - val_count) == 0 and total >= 2:
            train_count -= 1

        train_data.extend(vids[:train_count])
        val_data.extend(vids[train_count:train_count+val_count])
        test_data.extend(vids[train_count+val_count:])

    # Save to CSV
    def save_csv(data, filename):
        path = output_dir / filename
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_path", "class_name", "class_id"])
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved {len(data)} samples to {path}")

    save_csv(train_data, "train_split.csv")
    save_csv(val_data, "val_split.csv")
    save_csv(test_data, "test_split.csv")
    
    print("Dataset splitting complete.")

if __name__ == "__main__":
    main()
