import os, sys
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

@torch.no_grad()
def validate_and_find_threshold(model, val_dir, transform, device):
    """
    Validation function:
    1. Match the training forward pass exactly: Backbone -> LayerNorm -> L2 Normalize
    2. Calculate Top-1 identification accuracy
    3. Calculate cosine similarities for positive and negative pairs and use the ROC curve to find the optimal EER threshold
    """
    model.eval()
    
    # Support both DDP (model.module) and single-GPU (model) models.
    base_model = model.module if hasattr(model, 'module') else model
    backbone = base_model[0]
    layernorm = base_model[1]
    
    files = sorted([f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.png'))])
    orig_f, aug_f, orig_l, aug_l = [], [], [], []
    
    for f in files:
        img_path = os.path.join(val_dir, f)
        img = ImageOps.autocontrast(Image.open(img_path).convert('RGB'))
        tensor_img = transform(img).unsqueeze(0).to(device)
        
        # Match the training feature extraction pipeline exactly: Backbone -> LayerNorm -> L2 normalization.
        feat_raw = backbone(tensor_img)
        if isinstance(feat_raw, dict):  # Handle ViT variants that return a dictionary.
            feat_raw = feat_raw.get('x_norm_clstoken', feat_raw.get('x_norm', feat_raw))
        feat_norm = layernorm(feat_raw)
        feat = F.normalize(feat_norm, p=2, dim=1).cpu().numpy()
        
        pid = f.split('_')[0]
        if '_orig' in f:
            orig_f.append(feat)
            orig_l.append(pid)
        else:
            aug_f.append(feat)
            aug_l.append(pid)
            
    if not orig_f or not aug_f:
        return 0., 0.
        
    gallery = np.vstack(orig_f)  # Gallery features
    query = np.vstack(aug_f)      # Query features
    
    # Calculate the cosine similarity matrix [len(query), len(gallery)].
    sim = query @ gallery.T
    
    # 1. Calculate Top-1 accuracy.
    orig_l = np.array(orig_l)
    aug_l = np.array(aug_l)
    preds = orig_l[np.argmax(sim, axis=1)]
    acc = np.mean(preds == aug_l)
    
    # 2. Build positive and negative similarity pairs to calculate the EER threshold.
    pos_scores, neg_scores = [], []
    for i in range(len(query)):
        for j in range(len(gallery)):
            score = sim[i, j]
            if aug_l[i] == orig_l[j]:
                pos_scores.append(score)  # Image pairs from the same chip/class.
            else:
                neg_scores.append(score)  # Image pairs from different chips/classes.
                
    scores = np.array(pos_scores + neg_scores)
    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    
    # Calculate the ROC curve and equal error rate (EER) threshold.
    fpr, tpr, thr = roc_curve(labels, scores)
    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    best_threshold = thr[eer_idx]
    
    return acc, best_threshold

if __name__ == '__main__':
    # === Standalone testing/validation example ===
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    VAL_DIR = '/data/shurong/puf/puf_dataset/puf_splitted_dataset/val'
    REPO_PATH = '/data/shurong/puf/facebookresearch_dinov2_main'
    CKPT_PATH = '/data/shurong/puf/checkpoints/dinov2_classificationloss_best.pth'  # Best weights saved during training.
    
    if REPO_PATH not in sys.path:
        sys.path.append(REPO_PATH)
    from dinov2.models import vision_transformer as vits
    
    # 1. Rebuild the model architecture (must exactly match the training architecture).
    backbone = vits.__dict__['vit_base'](patch_size=14, img_size=518, init_values=1.0)
    
    # Assume the checkpoint contains the complete model state_dict.
    checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    
    # Remove the 'module.' prefix if the weights were saved by DDP.
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
        
    # Build the complete validation model.
    # Set num_classes to match your dataset; if available, it can be read from the checkpoint.
    num_classes = checkpoint.get('num_classes', 100) 
    model = nn.Sequential(
        backbone,
        nn.LayerNorm(768),
        nn.Dropout(0.4),
        nn.Linear(768, num_classes)
    ).to(device)
    
    model.load_state_dict(new_state_dict, strict=True)
    
    # Define the normalization transform used during validation.
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Run validation.
    acc, threshold = validate_and_find_threshold(model, VAL_DIR, transform, device)
    print(f"Validation Result -> Top-1 Accuracy: {acc:.2%}, EER Threshold: {threshold:.4f}")