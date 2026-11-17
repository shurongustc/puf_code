import os, sys, csv, shutil
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# === 1. Paths and configuration ===
REPO_PATH = '/data/shurong/puf/facebookresearch_dinov2_main'
TEST_DIR = '/data/shurong/puf/puf_dataset/puf_splitted_dataset/test'
CKPT_PATH = './checkpoints/dinov2_classificationloss_best.pth'  # Best weights saved during training.

# RESULT_DIR = './evaluation_results'
# WRONG_IMG_DIR = os.path.join(RESULT_DIR, 'wrong_cases')
# os.makedirs(WRONG_IMG_DIR, exist_ok=True)

# === 2. Feature extraction function (exactly matches training: Backbone -> LayerNorm -> L2 normalization) ===
@torch.no_grad()
def extract_features_with_metadata(data_dir, model, transform, device):
    model.eval()
    
    # Get the feature extraction components (backbone + LayerNorm).
    base_model = model.module if hasattr(model, 'module') else model
    backbone = base_model[0]
    layernorm = base_model[1]
    
    files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png'))])
    feats, labels, paths = [], [], []

    print(f"🔎 Processing test set: {os.path.basename(data_dir)}...")
    for f in tqdm(files):
        img_path = os.path.join(data_dir, f)
        img = ImageOps.autocontrast(Image.open(img_path).convert('RGB'))
        tensor_img = transform(img).unsqueeze(0).to(device)
        
        # Match the training validation forward pass.
        feat_raw = backbone(tensor_img)
        if isinstance(feat_raw, dict):
            feat_raw = feat_raw.get('x_norm_clstoken', feat_raw.get('x_norm', feat_raw))
        feat_norm = layernorm(feat_raw)
        feat = F.normalize(feat_norm, p=2, dim=1).cpu().numpy()
        
        label = f.split('_')[0]
        
        feats.append(feat)
        labels.append(label)
        paths.append(img_path)
            
    return np.vstack(feats), np.array(labels), paths

# === 3. Complete test evaluation workflow ===
def full_evaluation_workflow():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if REPO_PATH not in sys.path:
        sys.path.append(REPO_PATH)
    from dinov2.models import vision_transformer as vits

    # --- 3.1 Load the model ---
    print(f"📦 Loading model from {CKPT_PATH}...")
    backbone = vits.__dict__['vit_base'](patch_size=14, img_size=518, init_values=1.0)
    
    checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    num_classes = checkpoint.get('num_classes', 100)
    val_th = checkpoint.get('threshold', 0.0830)  # Best threshold saved during training validation.

    model = nn.Sequential(
        backbone,
        nn.LayerNorm(768),
        nn.Dropout(0.4),
        nn.Linear(768, num_classes)
    ).to(device)
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # --- 3.2 Extract features and split into Gallery (original images) and Query (augmented images) ---
    all_feats, all_labels, all_paths = extract_features_with_metadata(TEST_DIR, model, transform, device)
    
    g_idx = [i for i, p in enumerate(all_paths) if '_orig' in p]
    q_idx = [i for i, p in enumerate(all_paths) if '_aug' in p]
    
    g_feats, g_labels = all_feats[g_idx], all_labels[g_idx]
    q_feats, q_labels = all_feats[q_idx], all_labels[q_idx]
    g_paths = [all_paths[i] for i in g_idx]
    q_paths = [all_paths[i] for i in q_idx]

    # --- 3.3 Calculate the similarity matrix and Top-1 accuracy ---
    print("📊 Running retrieval analysis...")
    sim_matrix = q_feats @ g_feats.T  # Cosine similarity matrix [len(query), len(gallery)].
    
    best_match_indices = np.argmax(sim_matrix, axis=1)
    pred_labels = g_labels[best_match_indices]
    
    hits = np.sum(pred_labels == q_labels)
    top1_acc = hits / len(q_labels)

    # # Record misidentification cases and back up the images.
    # wrong_cases = []
    # for i in range(len(q_labels)):
    #     if pred_labels[i] != q_labels[i]:
    #         max_sim = sim_matrix[i, best_match_indices[i]]
    #         case = {
    #             'query_filename': os.path.basename(q_paths[i]),
    #             'true_id': q_labels[i],
    #             'pred_id': pred_labels[i],
    #             'similarity': f"{max_sim:.4f}",
    #             'gallery_match': os.path.basename(g_paths[best_match_indices[i]])
    #         }
    #         wrong_cases.append(case)
    #         save_name = f"ERR_True_{q_labels[i]}_Pred_{pred_labels[i]}.png"
    #         shutil.copy(q_paths[i], os.path.join(WRONG_IMG_DIR, save_name))

    # --- 3.4 Calculate evaluation metrics (TAR / FAR / ROC) ---
    label_mask = (q_labels[:, np.newaxis] == g_labels[np.newaxis, :])
    pos_scores = sim_matrix[label_mask]
    neg_scores = sim_matrix[~label_mask]
    
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_scores = np.concatenate([pos_scores, neg_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    tar = np.mean(pos_scores >= val_th)
    far = np.mean(neg_scores >= val_th)
    frr = 1.0 - tar

    # # --- 3.5 Plot and save the ROC curve ---
    # plt.figure(figsize=(8, 7))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'DINOv2 Fine-tuned ROC (AUC = {roc_auc:.4f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    # plt.scatter(far, tar, color='red', s=50, label=f'Threshold Point (th={val_th:.4f})', zorder=5)
    
    # plt.xlim([-0.01, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
    # plt.ylabel('True Acceptance Rate (TAR)', fontsize=12)
    # plt.title('PUF Identification ROC Curve (Test Set)', fontsize=14)
    # plt.legend(loc="lower right")
    # plt.grid(alpha=0.3)
    
    # roc_path = os.path.join(RESULT_DIR, 'roc_curve_test.png')
    # plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    # print(f"📈 ROC curve saved to: {roc_path}")

    # === 4. Print the final test report ===
    print("\n" + " Fine-tuned DINOv2 Classification Test Report ".center(45, "="))
    print(f"Total test samples (Query): {len(q_labels)}")
    print(f"Top-1 Accuracy: {top1_acc:.2%}")
    print(f"Validation threshold: {val_th:.4f}")
    print(f"True accept rate (TAR): {tar:.2%}")
    print(f"False accept rate (FAR): {far:.4%}")
    print(f"False reject rate (FRR): {frr:.2%}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("-" * 45)
    
    # csv_path = os.path.join(RESULT_DIR, 'test_analysis_report.csv')
    # with open(csv_path, 'w', newline='') as f:
    #     writer = csv.DictWriter(f, fieldnames=['query_filename', 'true_id', 'pred_id', 'similarity', 'gallery_match'])
    #     writer.writeheader()
    #     writer.writerows(wrong_cases)
    # print(f"📂 Detailed error report saved to: {csv_path}")
    # print("=" * 45)

if __name__ == '__main__':
    full_evaluation_workflow()