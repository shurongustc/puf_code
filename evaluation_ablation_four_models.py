import os, sys
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

REPO_PATH = '/data/shurong/puf/facebookresearch_dinov2_main'
VAL_DIR = '/data/shurong/puf/puf_dataset/puf_splitted_dataset/val'
TEST_DIR = '/data/shurong/puf/puf_dataset/puf_splitted_dataset/test'
RESULT_DIR = './evaluation_ablation_metrics'
os.makedirs(RESULT_DIR, exist_ok=True)
if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)
from dinov2.models import vision_transformer as vits

MODELS_TO_EVAL = [
    {'name':'PUF-DINO', 'ckpt':'./checkpoints/dinov2_puf_best_50epochs.pth', 'type':'retrieval', 'threshold':0.4809},
    {'name':'PUF-DINO-Cls', 'ckpt':'./checkpoints/dinov2_classificationloss_best.pth', 'type':'classification', 'threshold':0.0832},
    {'name':'DINO-Pretrained', 'ckpt':'./checkpoints/dino_vitb14_pretrain.pth', 'type':'pretrained', 'threshold':0.6676},
    {'name':'EffNet-V2L-Pretrained', 'ckpt':None, 'type':'effnet_pretrained', 'threshold':0.4956}
]

def load_dino(ckpt_path, model_type, device):
    if model_type == 'classification':
        backbone = vits.__dict__['vit_base'](
            patch_size=14, img_size=518, init_values=1.0
        )
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = {k.replace('module.',''): v for k, v in ckpt.get('model', ckpt).items()}
        num_classes = ckpt.get('num_classes', 100)
        model = nn.Sequential(
            backbone,
            nn.LayerNorm(768),
            nn.Dropout(0.4),
            nn.Linear(768, num_classes)
        ).to(device)
        model.load_state_dict(sd, strict=True)

    elif model_type == 'pretrained':
        model = vits.__dict__['vit_base'](
            patch_size=14, img_size=518,
            init_values=1.0, block_chunks=0
        )
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = {k.replace('module.',''): v for k, v in ckpt.get('model', ckpt).items()}
        model.load_state_dict(sd, strict=False)
        model = model.to(device)

    else:  # retrieval (PUF-DINO)
        model = vits.__dict__['vit_base'](
            patch_size=14, img_size=518,
            init_values=1.0
        )
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = {k.replace('module.',''): v for k, v in ckpt.get('model', ckpt).items()}
        model.load_state_dict(sd, strict=False)
        model = model.to(device)

    return model.eval()

@torch.no_grad()
def get_features(data_dir, model, model_type, transform, device):
    files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg','.png'))])
    of, af, ol, al = [], [], [], []
    for f in tqdm(files, desc=f'Extract {os.path.basename(data_dir)}', leave=False):
        img = ImageOps.autocontrast(Image.open(os.path.join(data_dir,f)).convert('RGB'))
        x = transform(img).unsqueeze(0).to(device)
        
        # ================= Core logic fixes =================
        if model_type == 'classification':
            # For classification models, extract features from the backbone (model[0]) only, avoiding logits from the classification head.
            out = model[0](x)
            if isinstance(out, dict):
                feat = out.get('x_norm_clstoken', out.get('x_norm', None))
                if feat is None:
                    feat = out['x_prenorm'][:, 0, :]
            else:
                # Use out[:, 0, :] to correctly extract the CLS token instead of out[0].
                feat = out[:, 0, :] if out.ndim == 3 else out
            
            # Apply LayerNorm manually if the Sequential contains it (i.e., model[1]).
            if len(model) > 1 and isinstance(model[1], nn.LayerNorm):
                feat = model[1](feat)
                
            feat = F.normalize(feat, p=2, dim=1)
        else:
            out = model(x)
            if isinstance(out, dict):
                feat = out['x_norm_clstoken']
            else:
                feat = out[:, 0, :] if out.ndim == 3 else out
            feat = F.normalize(feat, p=2, dim=1)
        # ================================================
            
        feat = feat.cpu().numpy()
        label = f.split('_')[0]
        if '_orig' in f: 
            of.append(feat)
            ol.append(label)
        elif '_aug' in f: 
            af.append(feat)
            al.append(label)
            
    return np.vstack(of), np.array(ol), np.vstack(af), np.array(al)

def get_roc_metrics(gf, gl, qf, ql, threshold=None):
    sims = qf @ gf.T
    mask = ql[:, None] == gl[None, :]
    pos, neg = sims[mask], sims[~mask]
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    fpr, tpr, ths = roc_curve(y, s)
    roc_auc = auc(fpr, tpr)
    
    if threshold is None:
        idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
        threshold = ths[idx]
        
    tar = np.mean(pos >= threshold)
    far = np.mean(neg >= threshold)
    return sims, roc_auc, fpr, tpr, tar, far, 1 - tar, threshold

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino_transform = transforms.Compose([
        transforms.Resize((518, 518)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    effnet_transform = transforms.Compose([
        transforms.Resize((384, 384)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    results = []

    for item in MODELS_TO_EVAL:
        name, ckpt, mtype = item['name'], item['ckpt'], item['type']
        transform = effnet_transform if mtype == 'effnet_pretrained' else dino_transform
        
        if mtype == 'effnet_pretrained':
            model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
            model.classifier = nn.Identity()
            model = model.to(device).eval()
            vg, vl, vq, vql = get_features(VAL_DIR, model, mtype, transform, device)
            _, val_auc, _, _, _, _, _, threshold = get_roc_metrics(vg, vl, vq, vql)
            print(f'\n[{name}] Val AUC: {val_auc:.4f}, EER threshold: {threshold:.4f}')
        else:
            model = load_dino(ckpt, mtype, device)
            threshold = item['threshold']

        tg, tl, tq, tql = get_features(TEST_DIR, model, mtype, transform, device)
        sims, roc_auc, fpr, tpr, tar, far, frr, threshold = get_roc_metrics(tg, tl, tq, tql, threshold)
        top1 = np.mean(tl[np.argmax(sims, axis=1)] == tql)

        print(f'[{name}] Top-1: {top1:.2%}, AUC: {roc_auc:.4f}, TAR: {tar:.2%}, FAR: {far:.4f}, FRR: {frr:.2%}, Threshold: {threshold:.4f}')
        results.append({'name': name, 'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr})

    plt.figure(figsize=(9, 7))
    for r in results:
        plt.plot(r['fpr'], r['tpr'], lw=2.5, label=f"{r['name']} (AUC={r['roc_auc']:.4f})")
    plt.xscale('log')
    plt.xlim([1e-4, 1.0])
    plt.ylim([0, 1.05])
    plt.xlabel('FAR', fontname='Arial', fontsize=18, fontweight='bold')
    plt.ylabel('TAR', fontname='Arial', fontsize=18, fontweight='bold')
    plt.xticks(fontname='Arial', fontsize=18, fontweight='bold')
    plt.yticks(fontname='Arial', fontsize=18, fontweight='bold')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    
    ax = plt.gca()
    for s in ax.spines.values(): 
        s.set_visible(True)
        s.set_linewidth(1.0)
        
    legend = plt.legend(loc='lower right', frameon=False)
    for text in legend.get_texts(): 
        text.set_fontname('Arial')
        text.set_fontsize(18)
        text.set_fontweight('bold')
        
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, 'four_models_comparison_roc.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f'\nROC saved to: {path}')

if __name__ == '__main__':
    evaluate()