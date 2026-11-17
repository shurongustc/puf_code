import os, sys
dinov2_source_path = "/data/shurong/puf/facebookresearch_dinov2_main"
if dinov2_source_path not in sys.path:
    sys.path.insert(0, dinov2_source_path)

import torch, numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from dinov2.models import vision_transformer as vits

# ================= Configuration =================
TEST_DIR = "/data/shurong/puf/puf_dataset/puf_splitted_dataset/test"
MODEL_PATH = "/data/shurong/puf/checkpoints/dinov2_puf_best_50epochs.pth"
CKPT_DINO = MODEL_PATH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 518 

print(f"Running on device: {DEVICE}")

# ================= Model loading =================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = vits.__dict__["vit_base"](patch_size=14, img_size=IMG_SIZE, init_values=1.0)
ckpt = torch.load(CKPT_DINO, map_location="cpu", weights_only=False)
sd = ckpt["model"] if "model" in ckpt else ckpt
sd = {k.replace("module.", ""): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.to(DEVICE).eval()
print("Model loaded and set to eval mode.")

# ================= Feature extraction function =================
def feature(path):
    try:
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            x = model(x)
            x = torch.nn.functional.normalize(x, dim=1)
        return x.cpu().numpy()[0]
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

# ================= Data loading and parsing =================
print(f"Reading images from {TEST_DIR}...")
puf_files_dict = {}
valid_extensions = (".jpg", ".png", ".jpeg", ".JPG", ".PNG")

for f in sorted(os.listdir(TEST_DIR)):
    if f.lower().endswith(valid_extensions):
        try:
            pid = f.split("_")[0]
            puf_files_dict.setdefault(pid, []).append(os.path.join(TEST_DIR, f))
        except IndexError:
            continue

puf_features_dict = {}
for pid, file_paths in tqdm(puf_files_dict.items(), desc="Extracting features"):
    features = []
    for fp in file_paths:
        feat = feature(fp)
        if feat is not None:
            features.append(feat)
    if len(features) > 1:
        puf_features_dict[pid] = np.stack(features)

print(f"Total PUF loaded with >1 images: {len(puf_features_dict)}")

if len(puf_features_dict) < 2:
    print("Error: Not enough PUF devices.")
    sys.exit(1)

# ================= Similarity calculation and statistics =================
print("Calculating similarity statistics...")
intra = []
for features in tqdm(puf_features_dict.values(), desc="Computing Intra-PUF"):
    s = cosine_similarity(features)
    n = len(features)
    intra.extend([s[i, j] for i in range(n) for j in range(i + 1, n)])
intra = np.array(intra)

inter = []
ids = list(puf_features_dict.keys())
for i in tqdm(range(len(ids)), desc="Computing Inter-PUF"):
    for j in range(i + 1, len(ids)):
        sim_matrix = cosine_similarity(puf_features_dict[ids[i]], puf_features_dict[ids[j]])
        inter.extend(sim_matrix.ravel())
inter = np.array(inter)

mu_intra, std_intra = intra.mean(), intra.std()
mu_inter, std_inter = inter.mean(), inter.std()
d_prime = np.abs(mu_intra - mu_inter) / np.sqrt(0.5 * (std_intra**2 + std_inter**2))
fdr = (mu_intra - mu_inter)**2 / (std_intra**2 + std_inter**2)

print(f"\n=== Statistics Summary ====")
print(f"Intra-PUF: Mean={mu_intra:.4f}, Std={std_intra:.4f}")
print(f"Inter-PUF: Mean={mu_inter:.4f}, Std={std_inter:.4f}")
print(f"Decidability Index (d'): {d_prime:.4f}")
print(f"Fisher Discriminant Ratio (FDR): {fdr:.4f}")
print(f"===========================\n")

# ================= Global plot settings (bold Arial, large font) =================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ================= 1. Plot the distribution histogram =================
plt.figure(figsize=(7, 5), dpi=300)
sns.histplot(inter, bins=100, stat="density", alpha=0.6, label="Inter-PUF", color="skyblue", edgecolor="white")
sns.histplot(intra, bins=100, stat="density", alpha=0.6, label="Intra-PUF", color="salmon", edgecolor="white")

plt.xlabel("Cosine Similarity", fontsize=18, fontweight='bold', fontname='Arial')
plt.ylabel("Density", fontsize=18, fontweight='bold', fontname='Arial')

plt.xticks(fontsize=16, fontname='Arial')
plt.yticks(fontsize=16, fontname='Arial')

legend = plt.legend(fontsize=16, frameon=True, facecolor='white', edgecolor='none')
for text in legend.get_texts():
    text.set_fontname('Arial')
    text.set_fontweight('bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("puf_uniqueness_distribution_v2.png", dpi=300)
plt.close()

# ================= 2. Plot the all-device similarity heatmap (Y-axis orientation fixed) =================
print("Generating clean correlation heatmap with customized index labels...")

puf_ids_ordered = sorted(list(puf_features_dict.keys()))
puf_representatives = np.array([np.mean(puf_features_dict[pid], axis=0) for pid in puf_ids_ordered])
full_sim_matrix = cosine_similarity(puf_representatives)
total_devices = len(puf_ids_ordered)

print(f"Full similarity matrix shape: {full_sim_matrix.shape}")

fig, ax_heatmap = plt.subplots(figsize=(8, 7), dpi=300)

im = sns.heatmap(
    full_sim_matrix,
    cmap="viridis",         
    vmin=0, vmax=1,         
    cbar=False,             
    square=True,            
    xticklabels=False,      
    yticklabels=False,      
    linewidths=0,           
    ax=ax_heatmap
)

# === Fixed: invert the Y-axis so index 0 is at the bottom and 300 is at the top. ===
ax_heatmap.invert_yaxis()

# Set ticks at 100, 200, and 300.
tick_positions = [100, 200, 300]
valid_ticks = [p for p in tick_positions if p <= total_devices]

ax_heatmap.set_xticks(valid_ticks)
ax_heatmap.set_xticklabels([str(p) for p in valid_ticks], fontsize=16, fontweight='bold', fontname='Arial')

ax_heatmap.set_yticks(valid_ticks)
ax_heatmap.set_yticklabels([str(p) for p in valid_ticks], fontsize=16, fontweight='bold', fontname='Arial', rotation=0)

# Clear the axis labels.
ax_heatmap.set_xlabel('')
ax_heatmap.set_ylabel('')

# Add an outer border to the heatmap.
for _, spine in ax_heatmap.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color('black')

# Add a colorbar.
cbar = fig.colorbar(im.collections[0], ax=ax_heatmap, orientation='vertical', fraction=0.046, pad=0.04, shrink=0.75)
cbar.set_label('Cosine Similarity', fontsize=18, fontweight='bold', fontname='Arial')
cbar.ax.tick_params(labelsize=16)
for label in cbar.ax.get_yticklabels():
    label.set_fontname('Arial')
    label.set_fontweight('bold')
cbar.outline.set_linewidth(1.5)

# Save the final figure.
heatmap_filename = "puf_correlation_heatmap_clean.png"
plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
plt.close('all')
print(f"Clean heatmap successfully saved as '{heatmap_filename}'")
print("Script finished successfully.")