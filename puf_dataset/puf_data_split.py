import os
import shutil
import random
from tqdm import tqdm

def split_puf_dataset(src_dir, dst_base, split_ratio=(0.8, 0.1, 0.1)):
    """
    src_dir: Source directory containing files named by ID.
    dst_base: Destination root directory.
    """
    # 1. Extract all unique IDs.
    # Assume filenames follow the format ID_xxx.jpg.
    all_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    ids = list(set([f.split('_')[0] for f in all_files]))
    random.shuffle(ids)

    # 2. Calculate the split point.
    n = len(ids)
    train_idx = int(n * split_ratio[0])
    val_idx = train_idx + int(n * split_ratio[1])

    id_splits = {
        'train': ids[:train_idx],
        'val': ids[train_idx:val_idx],
        'test': ids[val_idx:]
    }

    # 3. Move/copy the files.
    for mode, split_ids in id_splits.items():
        mode_dir = os.path.join(dst_base, mode)
        os.makedirs(mode_dir, exist_ok=True)
        print(f"📦 Processing {mode} split...")
        
        id_set = set(split_ids)
        for f in tqdm(all_files):
            if f.split('_')[0] in id_set:
                shutil.copy(os.path.join(src_dir, f), os.path.join(mode_dir, f))

# Example usage.
split_puf_dataset('/data/shurong/puf/puf_dataset/puf_dataset_augmented', '/data/shurong/puf/puf_dataset/puf_splitted_dataset')