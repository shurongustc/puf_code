import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

TEST_DIR = "/data/shurong/puf/puf_dataset/puf_splitted_dataset/test"
OUT_DIR = "./uniqueness_results_orig"
os.makedirs(OUT_DIR, exist_ok=True)

# Get all matching files.
files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith("_orig.jpg")])

names = []
vectors = []

# Load and flatten each original image.
for f in tqdm(files, desc="Loading PUFs (Original Size)"):
    img = cv2.imread(os.path.join(TEST_DIR, f), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Failed to read {f}, skipping.")
        continue
        
    # Flatten directly into a float32 vector to avoid doubling memory use or losing precision in later calculations.
    vectors.append(img.flatten().astype(np.float32))
    names.append(f.replace("_orig.jpg", ""))

# Convert to a large (N, D) matrix; 300 original images use about 24 GB of memory.
vectors = np.stack(vectors)
print(f"Number of PUFs: {len(names)}")
print(f"Feature matrix shape: {vectors.shape}")

# Calculate the Pearson correlation matrix (N, N).
print("Computing correlation matrix...")
corr = np.corrcoef(vectors)

# Save the correlation matrix.
np.save(os.path.join(OUT_DIR, "correlation_matrix_orig.npy"), corr)

# 1. Plot the heatmap.
plt.figure(figsize=(9, 8), dpi=300)
im = plt.imshow(corr, cmap="viridis", vmin=0, vmax=1)
plt.colorbar(im, label="Pearson Correlation")
plt.title(f"Panoramic Correlation Matrix of {len(names)} LC-PUF Devices (Orig)", fontsize=16, pad=20)
plt.xlabel("PUF Device Index", fontsize=14, fontweight="bold")
plt.ylabel("PUF Device Index", fontsize=14, fontweight="bold")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pearson_correlation_matrix_orig.png"), bbox_inches="tight")
plt.close()

# Extract the upper triangular matrix (excluding the diagonal self-correlations of 1.0).
upper = corr[np.triu_indices_from(corr, k=1)]

# 2. Plot the histogram.
plt.figure(figsize=(6, 4), dpi=300)
plt.hist(upper, bins=60, density=True)
plt.xlabel("Pearson Correlation", fontsize=12, fontweight="bold")
plt.ylabel("Density", fontsize=12, fontweight="bold")
plt.title("Inter-PUF Correlation Distribution (Orig)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "inter_puf_histogram_orig.png"), bbox_inches="tight")
plt.close()

# Print the statistics.
print("-" * 30)
print(f"Mean: {upper.mean():.4f}")
print(f"Std : {upper.std():.4f}")
print(f"Min : {upper.min():.4f}")
print(f"Max : {upper.max():.4f}")
print("-" * 30)
print(f"Results saved to {OUT_DIR}")