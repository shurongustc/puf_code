import os
import numpy as np
import matplotlib.pyplot as plt

# Set the global font to bold Arial, size 18.
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

OUT_DIR = "./uniqueness_results_orig"
corr_path = os.path.join(OUT_DIR, "correlation_matrix_orig.npy")

# Load the previously saved NPY file.
corr = np.load(corr_path)
n_devices = corr.shape[0]

plt.figure(figsize=(9, 8), dpi=300)

# --- Key change ---
# Set origin='lower' to place row 0 at the bottom and row N-1 at the top.
im = plt.imshow(corr, cmap="viridis", vmin=0, vmax=1, origin='lower')
# --- End of key change ---

# Shorten the colorbar on the right.
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label("Pearson Correlation", fontsize=18, fontweight="bold")
for label in cbar.ax.get_yticklabels():
    label.set_fontname('Arial')
    label.set_fontsize(16)
    label.set_weight('bold')

# Axis labels.
idx_label = f"PUF Device Index (0 - {n_devices - 1})"
plt.xlabel(idx_label, fontsize=18, fontweight="bold", labelpad=10)
plt.ylabel(idx_label, fontsize=18, fontweight="bold", labelpad=10)

# Tick label font.
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

plt.tight_layout()

# Save the new figure.
output_path = os.path.join(OUT_DIR, "correlation_matrix_custom_fixed.png")
plt.savefig(output_path, bbox_inches="tight")
plt.close()

print(f"Heatmap with y-axis increasing from bottom saved to {output_path}")