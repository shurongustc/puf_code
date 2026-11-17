import os, sys
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# === 1. Paths & Configuration ===
REPO_PATH = '/data/shurong/puf/facebookresearch_dinov2_main'
TEST_DIR = '/data/shurong/puf/puf_dataset/puf_splitted_dataset/test'
RESULT_DIR = './evaluation_ablation_metrics'
os.makedirs(RESULT_DIR, exist_ok=True)

if REPO_PATH not in sys.path: 
    sys.path.append(REPO_PATH)
from dinov2.models import vision_transformer as vits

MODELS_TO_EVAL = [
    {
        'name': 'PUF-DINO',
        'ckpt': './checkpoints/dinov2_puf_best_50epochs.pth',
        'type': 'retrieval',
        'threshold': 0.4809
    },
    {
        'name': 'PUF-DINO-Cls',
        'ckpt': './checkpoints/dinov2_classificationloss_best.pth',
        'type': 'classification',
        'threshold': 0.0832
    },
    {
        'name': 'DINO-Pretrained',
        'ckpt': './checkpoints/dino_vitb14_pretrain.pth', 
        'type': 'pretrained',
        'threshold': 0.6676
    }
]

# === 2. Feature Extraction Function (Tailored per Model Type) ===
@torch.no_grad()
def get_features(data_dir, model, model_type, transform, device):
    model.eval()
    files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png'))])
    orig_f, aug_f, orig_l, aug_l = [], [], [], []
    
    print(f"📦 Extracting features for {os.path.basename(data_dir)} ({model_type})...")
    for f in tqdm(files, leave=False):
        img_path = os.path.join(data_dir, f)
        img = ImageOps.autocontrast(Image.open(img_path).convert('RGB'))
        tensor_img = transform(img).unsqueeze(0).to(device)
        
        if model_type == 'classification':
            base_model = model.module if hasattr(model, 'module') else model
            backbone = base_model[0]
            layernorm = base_model[1]
            feat_raw = backbone(tensor_img)
            if isinstance(feat_raw, dict):
                feat_raw = feat_raw.get('x_norm_clstoken', feat_raw.get('x_norm', feat_raw))
            feat_norm = layernorm(feat_raw)
            feat = F.normalize(feat_norm, p=2, dim=1).cpu().numpy()
        elif model_type == 'pretrained':
            out = model(tensor_img)
            feat = out["x_norm_clstoken"] if isinstance(out, dict) else out
            feat = torch.nn.functional.normalize(feat, p=2, dim=1).cpu().numpy()
        else:  # retrieval (PUF-DINO)
            out = model(tensor_img)
            feat = out["x_norm_clstoken"] if isinstance(out, dict) else out
            feat = torch.nn.functional.normalize(feat, p=2, dim=1).cpu().numpy()
            
        label = f.split('_')[0]
        if '_orig' in f:
            orig_f.append(feat)
            orig_l.append(label)
        elif '_aug' in f:
            aug_f.append(feat)
            aug_l.append(label)
            
    return np.vstack(orig_f), np.array(orig_l), np.vstack(aug_f), np.array(aug_l)

# === 3. Main Evaluation & ROC Plotting ===
def evaluate_and_plot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    results = []

    for item in MODELS_TO_EVAL:
        name = item['name']
        ckpt_path = item['ckpt']
        m_type = item['type']
        eval_threshold = item['threshold']
        
        print(f"\n--- Evaluating Model: {name} ---")
        print(f"📌 Using threshold: {eval_threshold:.4f}")
        
        if m_type == 'classification':
            backbone = vits.__dict__['vit_base'](patch_size=14, img_size=518, init_values=1.0)
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('model', checkpoint)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            num_classes = checkpoint.get('num_classes', 100)
            
            model = nn.Sequential(
                backbone,
                nn.LayerNorm(768),
                nn.Dropout(0.4),
                nn.Linear(768, num_classes)
            ).to(device)
            model.load_state_dict(state_dict, strict=True)
        elif m_type == 'pretrained':
            model = vits.__dict__['vit_base'](patch_size=14, img_size=518, init_values=1.0, block_chunks=0)
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
        else:  # retrieval
            model = vits.__dict__['vit_base'](patch_size=14, img_size=518, init_values=1.0)
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model.to(device)

        model.eval()

        # Test set evaluation
        t_gal_f, t_gal_l, t_que_f, t_que_l = get_features(TEST_DIR, model, m_type, transform, device)
        sims = t_que_f @ t_gal_f.T

        pred = np.argmax(sims, axis=1)
        top1 = np.mean(t_gal_l[pred] == t_que_l)

        mask = t_que_l[:, None] == t_gal_l[None, :]
        pos = sims[mask]
        neg = sims[~mask]

        tar = np.mean(pos >= eval_threshold)
        far = np.mean(neg >= eval_threshold)
        frr = 1 - tar

        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        s = np.concatenate([pos, neg])
        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = auc(fpr, tpr)

        print(f"[{name}] Top-1: {top1:.2%}, AUC: {roc_auc:.4f}, TAR: {tar:.2%}, FAR: {far:.4f}, FRR: {frr:.2%}")

        results.append({
            'name': name,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'tar_at_th': tar,
            'far_at_th': far
        })

    # Plot multi-model comparison ROC-AUC curve with top/right borders enabled
    plt.figure(figsize=(9, 7))
    for r in results:
        plt.plot(r['fpr'], r['tpr'], lw=2.5, label=f"{r['name']} (AUC={r['roc_auc']:.4f})")
    
    plt.xscale('log')
    plt.xlim([1e-4, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('FAR', fontname='Arial', fontsize=18, fontweight='bold')
    plt.ylabel('TAR', fontname='Arial', fontsize=18, fontweight='bold')
    
    plt.xticks(fontname='Arial', fontsize=18, fontweight='bold')
    plt.yticks(fontname='Arial', fontsize=18, fontweight='bold')
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Enable top and right borders to form a fully enclosed box
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['top'].set_linewidth(1.0)
    plt.gca().spines['right'].set_linewidth(1.0)
    plt.gca().spines['bottom'].set_linewidth(1.0)
    plt.gca().spines['left'].set_linewidth(1.0)
    
    legend = plt.legend(loc="lower right", frameon=False)
    for text in legend.get_texts():
        text.set_fontname('Arial')
        text.set_fontsize(18)
        text.set_fontweight('bold')
        
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, 'three_models_comparison_roc.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\n✨ Multi-model comparison ROC curve successfully generated and saved to: {save_path}")

if __name__ == "__main__":
    evaluate_and_plot()