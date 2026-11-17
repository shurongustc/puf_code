import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torchvision.transforms as transforms

# --- Parameter settings ---
IMG_SIZE = 518
PATCH_SIZE = 14
GRID_SIZE = IMG_SIZE // PATCH_SIZE # 37

def get_patch_features(model, img_t):
    """Extract DINOv2 patch features (Batch, 37*37, Dim)."""
    with torch.no_grad():
        # DINOv2 returns features from the last four layers or a specified layer.
        # Extract the patch tokens from the last layer here.
        intermediate = model.get_intermediate_layers(img_t, n=1)[0]
        # Normalize the features.
        return torch.nn.functional.normalize(intermediate, p=2, dim=-1)

def visualize_alignment(query_path, gallery_path, model, transform, device, save_name="heatmap.png"):
    model.eval()
    
    # 1. Load and transform the image.
    q_raw = Image.open(query_path).convert('RGB')
    g_raw = Image.open(gallery_path).convert('RGB')
    
    q_img = transform(q_raw).unsqueeze(0).to(device)
    g_img = transform(g_raw).unsqueeze(0).to(device)

    # 2. Extract patch features.
    q_feat = get_patch_features(model, q_img) # [1, 1369, 768]
    g_feat = get_patch_features(model, g_img) # [1, 1369, 768]

    # 3. Calculate the cosine similarity matrix [1369, 1369].
    # Each entry represents the similarity between a Query patch and a Gallery patch.
    sim_matrix = torch.bmm(q_feat, g_feat.transpose(1, 2)).squeeze(0).cpu().numpy()

    # 4. Generate the heatmap.
    # Identify which Query patches find strong matches in the Gallery.
    # Take the maximum response of each Query patch across the entire Gallery.
    heatmap_data = sim_matrix.max(axis=1).reshape(GRID_SIZE, GRID_SIZE)
    
    # Normalize for display.
    heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-8)

    # 5. Plot the results.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display the original image.
    axes[0].imshow(q_raw.resize((IMG_SIZE, IMG_SIZE)))
    axes[0].set_title("Query Image (PUF Aug/Test)", fontsize=12)
    
    axes[1].imshow(g_raw.resize((IMG_SIZE, IMG_SIZE)))
    axes[1].set_title("Gallery Image (PUF Orig)", fontsize=12)

    # Overlay the heatmap.
    axes[2].imshow(q_raw.resize((IMG_SIZE, IMG_SIZE)))
    # Use BICUBIC interpolation to smooth the heatmap and reduce blockiness.
    extent = (0, IMG_SIZE, IMG_SIZE, 0)
    im = axes[2].imshow(heatmap_norm, cmap='jet', alpha=0.5, extent=extent, interpolation='bicubic')
    axes[2].set_title("Physical Feature Alignment Map", fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=200)
    print(f"🔥 Heatmap saved to: {save_name}")

# --- Main program ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the environment and model (following the existing loading logic).
    REPO_PATH = '/data/shurong/puf/facebookresearch_dinov2_main'
    CKPT_PATH = './dinov2_puf_best.pth'
    if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)
    from dinov2.models import vision_transformer as vits

    model = vits.__dict__['vit_base'](patch_size=14, img_size=518, init_values=1.0)
    checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    sd = checkpoint['model'] if 'model' in checkpoint else checkpoint
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Specify the paths of the two images to visualize.
    Q_PATH = "test_data/ID001_aug_01.png"
    G_PATH = "test_data/ID001_orig.png"
    
    visualize_alignment(Q_PATH, G_PATH, model, transform, device, "puf_alignment_analysis.png")