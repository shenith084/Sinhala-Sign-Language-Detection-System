import os
import pandas as pd

def create_top50_subset(data_dir="data/splits"):
    print("Loading original splits...")
    train_df = pd.read_csv(os.path.join(data_dir, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val_split.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_split.csv"))
    word_map = pd.read_csv(os.path.join(data_dir, "sinhala_word_map.csv"))

    # Find Top 50 classes based on training frequency
    counts = train_df['class_name'].value_counts()
    top50_classes = counts.head(50).index.tolist()
    
    print(f"Top 50 classes identified. Examples: {top50_classes[:5]}")

    # Create new class ID mapping (0 to 49)
    new_mapping = {class_name: idx for idx, class_name in enumerate(top50_classes)}

    def filter_and_remap(df):
        # Filter only top 50
        filtered = df[df['class_name'].isin(top50_classes)].copy()
        # Remap IDs
        filtered['class_id'] = filtered['class_name'].map(new_mapping)
        return filtered

    # Process all splits
    print("Filtering and remapping datasets...")
    train_50 = filter_and_remap(train_df)
    val_50 = filter_and_remap(val_df)
    test_50 = filter_and_remap(test_df)

    # Process word map
    word_map_50 = word_map[word_map['word'].isin(top50_classes)].copy()
    word_map_50['class_id'] = word_map_50['word'].map(new_mapping)
    word_map_50 = word_map_50.sort_values('class_id')

    # Save to disk (overwriting the old ones to enforce the 50-class subset globally)
    train_50.to_csv(os.path.join(data_dir, "train_split.csv"), index=False)
    val_50.to_csv(os.path.join(data_dir, "val_split.csv"), index=False)
    test_50.to_csv(os.path.join(data_dir, "test_split.csv"), index=False)
    word_map_50.to_csv(os.path.join(data_dir, "sinhala_word_map.csv"), index=False)

    print(f"Success! Saved Train: {len(train_50)}, Val: {len(val_50)}, Test: {len(test_50)}")
    print("WARNING: You must update config.yaml dataset.num_classes to 50!")

if __name__ == "__main__":
    create_top50_subset()
