import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Set the global font.
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

TEST_DIR = "/data/shurong/puf/puf_dataset/puf_splitted_dataset/test"
OUT_DIR = "./uniqueness_results_orig"
os.makedirs(OUT_DIR, exist_ok=True)

all_files = sorted(os.listdir(TEST_DIR))

# 1. Extract the _orig images used for Inter-PUF.
orig_files = [f for f in all_files if f.endswith("_orig.jpg")]
rep_vectors = []
for f in orig_files:
    img = cv2.imread(os.path.join(TEST_DIR, f), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        rep_vectors.append(img.flatten().astype(np.float32))

rep_vectors = np.stack(rep_vectors)
inter_corr_matrix = np.corrcoef(rep_vectors)
inter_corrs = inter_corr_matrix[np.triu_indices_from(inter_corr_matrix, k=1)]

# 2. Extract the _aug images used for Intra-PUF, grouped by device ID.
aug_files = [f for f in all_files if "_aug" in f]  # Match your augmented image filenames.
aug_dict = defaultdict(list)
for f in aug_files:
    # Expected filename formats: deviceID_xxx_aug.jpg or deviceID_aug1.jpg.
    # Extract dev_id according to your filenames, e.g., the text before the first underscore.
    dev_id = f.split("_")[0] 
    aug_dict[dev_id].append(f)

intra_corrs = []
for dev_id, sample_files in aug_dict.items():
    if len(sample_files) < 2:
        continue  # Intra-device correlation cannot be calculated if a device has only one augmented image.
    
    vecs = []
    for sf in sample_files:
        img = cv2.imread(os.path.join(TEST_DIR, sf), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            vecs.append(img.flatten().astype(np.float32))
    
    if len(vecs) >= 2:
        vecs = np.stack(vecs)
        sub_corr = np.corrcoef(vecs)
        upper_sub = sub_corr[np.triu_indices_from(sub_corr, k=1)]
        intra_corrs.extend(upper_sub)

intra_corrs = np.array(intra_corrs)

# 3. Plot the comparison histogram.
plt.figure(figsize=(7, 5), dpi=300)

plt.hist(inter_corrs, bins=50, density=True, alpha=0.6, color='blue', label='Inter-PUF (Orig)')
if len(intra_corrs) > 0:
    plt.hist(intra_corrs, bins=50, density=True, alpha=0.6, color='red', label='Intra-PUF (Aug)')

plt.xlabel("Pearson Correlation", fontsize=18, fontweight="bold", labelpad=10)
plt.ylabel("Density", fontsize=18, fontweight="bold", labelpad=10)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(frameon=False, prop={'family': 'Arial', 'size': 16, 'weight': 'bold'})

plt.tight_layout()
output_path = os.path.join(OUT_DIR, "inter_orig_vs_intra_aug_histogram.png")
plt.savefig(output_path, bbox_inches="tight")
plt.close()

print(f"Histogram saved to {output_path}")
print(f"Inter-PUF (Orig) Mean: {inter_corrs.mean():.4f}, Std: {inter_corrs.std():.4f}")
if len(intra_corrs) > 0:
    print(f"Intra-PUF (Aug) Mean: {intra_corrs.mean():.4f}, Std: {intra_corrs.std():.4f}")