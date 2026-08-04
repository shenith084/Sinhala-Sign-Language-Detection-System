import os
import pandas as pd
from pathlib import Path

def create_top10_subset(data_dir="data/splits"):
    print("Loading original splits...")
    train_df = pd.read_csv(os.path.join(data_dir, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val_split.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_split.csv"))
    word_map = pd.read_csv(os.path.join(data_dir, "sinhala_word_map.csv"))

    # Find Top 10 classes based on training frequency
    counts = train_df['class_name'].value_counts()
    top10_classes = counts.head(10).index.tolist()
    
    print(f"Top 10 classes identified: {top10_classes}")

    # Create new class ID mapping (0 to 9)
    new_mapping = {class_name: idx for idx, class_name in enumerate(top10_classes)}

    def filter_and_remap(df):
        # Filter only top 10
        filtered = df[df['class_name'].isin(top10_classes)].copy()
        # Remap IDs
        filtered['class_id'] = filtered['class_name'].map(new_mapping)
        return filtered

    # Process all splits
    print("Filtering and remapping datasets...")
    train_10 = filter_and_remap(train_df)
    val_10 = filter_and_remap(val_df)
    test_10 = filter_and_remap(test_df)

    # Process word map using class_name_english
    word_map_10 = word_map[word_map['class_name_english'].isin(top10_classes)].copy()
    word_map_10['class_id'] = word_map_10['class_name_english'].map(new_mapping)
    word_map_10 = word_map_10.sort_values('class_id')

    # Save to disk (overwriting the old ones to enforce the 10-class subset globally)
    train_10.to_csv(os.path.join(data_dir, "train_split.csv"), index=False)
    val_10.to_csv(os.path.join(data_dir, "val_split.csv"), index=False)
    test_10.to_csv(os.path.join(data_dir, "test_split.csv"), index=False)
    word_map_10.to_csv(os.path.join(data_dir, "sinhala_word_map.csv"), index=False)

    print(f"Success! Saved Train: {len(train_10)}, Val: {len(val_10)}, Test: {len(test_10)}")
    print("Successfully mapped Sinhala class names for the Top 10 words!")

if __name__ == "__main__":
    create_top10_subset()
