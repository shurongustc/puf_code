import os, sys, csv, shutil
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_l
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import torch

TEST_DIR = '/data/shurong/puf/puf_dataset/puf_splitted_dataset/test'
CKPT_PATH = './checkpoints/effnet_v2l_best.pth'

# Enter the threshold value you recall here.
THRESHOLD = 0.4956

RESULT_DIR = './effnet_v2l_evaluation_results'
WRONG_IMG_DIR = os.path.join(RESULT_DIR, 'wrong_cases')
os.makedirs(WRONG_IMG_DIR, exist_ok=True)

def load_model(path, device):
    model = efficientnet_v2_l(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_f, 768),
        torch.nn.LayerNorm(768),
        torch.nn.Dropout(0.4)
    )
    ckpt = torch.load(path, map_location='cpu')
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(device).eval()

@torch.no_grad()
def extract_features(data_dir, model, transform, device):
    feats, labels, paths = [], [], []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))])
    for f in tqdm(files, desc='Extract'):
        img_path = os.path.join(data_dir, f)
        img = ImageOps.autocontrast(Image.open(img_path).convert('RGB'))
        x = transform(img).unsqueeze(0).to(device)
        feat = torch.nn.functional.normalize(model(x), dim=1).cpu().numpy()
        feats.append(feat)
        labels.append(f.split('_')[0])
        paths.append(img_path)
    return np.vstack(feats), np.array(labels), paths

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(CKPT_PATH, device)
    
    # Use the manually specified threshold.
    threshold = THRESHOLD
    print(f"Using validation threshold: {threshold:.4f}")

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    feats, labels, paths = extract_features(TEST_DIR, model, transform, device)

    g = np.array(['_orig' in p for p in paths])
    q = np.array(['_aug' in p for p in paths])

    gf, gl = feats[g], labels[g]
    qf, ql = feats[q], labels[q]
    g_paths = [paths[i] for i, is_g in enumerate(g) if is_g]
    q_paths = [paths[i] for i, is_q in enumerate(q) if is_q]

    sim = qf @ gf.T

    best_match_indices = np.argmax(sim, axis=1)
    pred_labels = gl[best_match_indices]
    top1 = np.mean(pred_labels == ql)

    # Record error cases.
    wrong_cases = []
    for i in range(len(ql)):
        if pred_labels[i] != ql[i]:
            max_sim = sim[i, best_match_indices[i]]
            case = {
                'query_filename': os.path.basename(q_paths[i]),
                'true_id': ql[i],
                'pred_id': pred_labels[i],
                'similarity': f"{max_sim:.4f}",
                'gallery_match': os.path.basename(g_paths[best_match_indices[i]])
            }
            wrong_cases.append(case)
            save_name = f"ERR_True_{ql[i]}_Pred_{pred_labels[i]}_{os.path.basename(q_paths[i])}"
            shutil.copy(q_paths[i], os.path.join(WRONG_IMG_DIR, save_name))

    pos = sim[ql[:, None] == gl[None, :]]
    neg = sim[ql[:, None] != gl[None, :]]

    tar = np.mean(pos >= threshold)
    far = np.mean(neg >= threshold)
    frr = 1 - tar

    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])

    fpr, tpr, _ = roc_curve(y, s)
    auc_score = auc(fpr, tpr)

    print("\n" + "=" * 60)
    print(f"{'Model':<15}{'Top1':>10}{'AUC':>10}{'TAR':>10}{'FAR':>10}{'FRR':>10}")
    print("-" * 60)
    print(f"{'EffNet-V2L':<15}{top1:>10.2%}{auc_score:>10.4f}{tar:>10.2%}{far:>10.4%}{frr:>10.2%}")
    print("-" * 60)
    print(f"Threshold: {threshold:.4f}")

    # Save the CSV report.
    csv_path = os.path.join(RESULT_DIR, 'test_wrong_cases_report.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['query_filename', 'true_id', 'pred_id', 'similarity', 'gallery_match'])
        writer.writeheader()
        writer.writerows(wrong_cases)
    print(f"📂 Wrong cases report saved to: {csv_path}")
    print(f"📂 Wrong case images copied to: {WRONG_IMG_DIR}")

if __name__ == '__main__':
    evaluate()