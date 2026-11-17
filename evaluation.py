import torch
import os
import sys
import numpy as np
import shutil
import csv
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

# === 1. Path and environment configuration ===
REPO_PATH = '/data/shurong/puf/facebookresearch_dinov2_main'
TEST_DIR  = '/data/shurong/puf/puf_dataset/puf_splitted_dataset/test'
#CKPT_PATH = './checkpoints/dinov2_puf_best.pth' 
CKPT_PATH = './checkpoints/dinov2_puf_best_50epochs.pth'

RESULT_DIR = './evaluation_results'
WRONG_IMG_DIR = os.path.join(RESULT_DIR, 'wrong_cases')
os.makedirs(WRONG_IMG_DIR, exist_ok=True)

if REPO_PATH not in sys.path: 
    sys.path.append(REPO_PATH)
from dinov2.models import vision_transformer as vits

# === 2. Feature extraction function ===
@torch.no_grad()
def extract_features_with_metadata(data_dir, model, transform, device):
    model.eval()
    files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png'))])
    
    feats, labels, paths = [], [], []

    print(f"🔎 Processing dataset: {os.path.basename(data_dir)}...")
    for f in tqdm(files):
        img_path = os.path.join(data_dir, f)
        # Preprocessing: match the training pipeline.
        img = ImageOps.autocontrast(Image.open(img_path).convert('RGB'))
        img_t = transform(img).unsqueeze(0).to(device)
        
        # Extract features and apply L2 normalization.
        output = model(img_t)
        feat = torch.nn.functional.normalize(output, p=2, dim=1).cpu().numpy()
        
        # Expected filename format: ID_type.png (e.g., 101_orig.png).
        label = f.split('_')[0]
        
        feats.append(feat)
        labels.append(label)
        paths.append(img_path)
            
    return np.vstack(feats), np.array(labels), paths

# === 3. Core evaluation and visualization workflow ===
def full_evaluation_workflow():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 3.1 Load the model ---
    print(f"📦 Loading model from {CKPT_PATH}...")
    model = vits.__dict__['vit_base'](patch_size=14, img_size=518, init_values=1.0)
    checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    # Get the best threshold from training, or use the default if unavailable.
    val_th = checkpoint.get('threshold', 0.45) 

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # --- 3.2 Extract features and split into Gallery/Query ---
    all_feats, all_labels, all_paths = extract_features_with_metadata(TEST_DIR, model, transform, device)
    
    g_idx = [i for i, p in enumerate(all_paths) if '_orig' in p]
    q_idx = [i for i, p in enumerate(all_paths) if '_aug' in p]
    
    g_feats, g_labels = all_feats[g_idx], all_labels[g_idx]
    q_feats, q_labels = all_feats[q_idx], all_labels[q_idx]
    g_paths = [all_paths[i] for i in g_idx]
    q_paths = [all_paths[i] for i in q_idx]

    # --- 3.3 Calculate the similarity matrix and Top-1 accuracy ---
    print("📊 Running retrieval analysis...")
    sim_matrix = q_feats @ g_feats.T  # Cosine similarity matrix.
    
    # Calculate Top-1 matches.
    best_match_indices = np.argmax(sim_matrix, axis=1)
    pred_labels = g_labels[best_match_indices]
    
    hits = np.sum(pred_labels == q_labels)
    top1_acc = hits / len(q_labels)

    # Record error cases.
    wrong_cases = []
    for i in range(len(q_labels)):
        if pred_labels[i] != q_labels[i]:
            max_sim = sim_matrix[i, best_match_indices[i]]
            case = {
                'query_filename': os.path.basename(q_paths[i]),
                'true_id': q_labels[i],
                'pred_id': pred_labels[i],
                'similarity': f"{max_sim:.4f}",
                'gallery_match': os.path.basename(g_paths[best_match_indices[i]])
            }
            wrong_cases.append(case)
            # Copy images for later analysis.
            save_name = f"ERR_True_{q_labels[i]}_Pred_{pred_labels[i]}.png"
            shutil.copy(q_paths[i], os.path.join(WRONG_IMG_DIR, save_name))

    # --- 3.4 Calculate validation metrics (TAR/FAR/ROC) ---
    # Build a mask: True for positive pairs and False for negative pairs.
    label_mask = (q_labels[:, np.newaxis] == g_labels[np.newaxis, :])
    
    pos_scores = sim_matrix[label_mask]
    neg_scores = sim_matrix[~label_mask]
    
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_scores = np.concatenate([pos_scores, neg_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate performance at the specified threshold.
    tar = np.mean(pos_scores >= val_th)
    far = np.mean(neg_scores >= val_th)
    frr = 1.0 - tar

    # --- 3.5 Plot the ROC curve ---
    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'DINOv2 ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.scatter(far, tar, color='red', s=50, label=f'Operating Point (th={val_th})', zorder=5)
    
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
    plt.ylabel('True Acceptance Rate (TAR)', fontsize=12)
    plt.title('ROC Curve for PUF Identification', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    roc_path = os.path.join(RESULT_DIR, 'roc_curve_test.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"📈 ROC curve saved to: {roc_path}")

    # === 4. Print the final report ===
    print("\n" + " Fine-tuned DINOv2 Performance Report ".center(45, "="))
    print(f"Total test samples: {len(q_labels)}")
    print(f"Top-1 Accuracy: {top1_acc:.2%}")
    print(f"Validation threshold: {val_th:.4f}")
    print(f"True accept rate (TAR): {tar:.2%}")
    print(f"False accept rate (FAR): {far:.4%}")
    print(f"False reject rate (FRR): {frr:.2%}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("-" * 45)
    
    # Save the CSV.
    csv_path = os.path.join(RESULT_DIR, 'test_analysis_report.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['query_filename', 'true_id', 'pred_id', 'similarity', 'gallery_match'])
        writer.writeheader()
        writer.writerows(wrong_cases)
    print(f"📂 Detailed error report saved to: {csv_path}")
    print("=" * 45)

if __name__ == "__main__":
    full_evaluation_workflow()
