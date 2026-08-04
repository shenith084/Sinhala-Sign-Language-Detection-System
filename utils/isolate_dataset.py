import os
import shutil
import csv
import collections

def isolate_top_50():
    base_dir = "c:/project/ssl400_research_project"
    original_dir = os.path.join(base_dir, "SSL400", "Dataset - Original")
    excluded_dir = os.path.join(base_dir, "SSL400", "Dataset - Excluded")
    splits_dir = os.path.join(base_dir, "data", "splits")
    train_csv = os.path.join(splits_dir, "train_split.csv")

    if not os.path.exists(train_csv):
        print(f"Error: Could not find {train_csv}. Have you run generate_splits.py before?")
        return

    print("Loading class frequencies...")
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        counts = collections.Counter(row[1] for row in reader)
        
    top_50 = [item[0] for item in counts.most_common(50)]
    
    print(f"Keeping top 50 classes. Example: {top_50[:5]}")
    
    # Create Excluded directory
    os.makedirs(excluded_dir, exist_ok=True)

    moved_count = 0
    kept_count = 0

    # Iterate over Categories (Adjectives, Nouns, etc.)
    for category in os.listdir(original_dir):
        cat_path = os.path.join(original_dir, category)
        if not os.path.isdir(cat_path): continue

        # Iterate over Words (Hello, House, etc.)
        for word in os.listdir(cat_path):
            word_path = os.path.join(cat_path, word)
            if not os.path.isdir(word_path): continue

            if word not in top_50:
                # Move to excluded
                dest_cat_dir = os.path.join(excluded_dir, category)
                os.makedirs(dest_cat_dir, exist_ok=True)
                dest_word_dir = os.path.join(dest_cat_dir, word)
                
                print(f"Excluding: {category}/{word}")
                shutil.move(word_path, dest_word_dir)
                moved_count += 1
            else:
                kept_count += 1

    print("\n--- Summary ---")
    print(f"Classes Kept: {kept_count} (Should be 50)")
    print(f"Classes Excluded: {moved_count}")
    print(f"The excluded videos have been safely moved to {excluded_dir}")
    print("\nYou can now delete your data/splits and data/processed folders, and run generate_splits.py to start fresh with a perfect 50-class dataset!")

if __name__ == "__main__":
    isolate_top_50()
