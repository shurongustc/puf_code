import os, csv, shutil
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import torch

VAL_DIR = '/data/shurong/puf/puf_dataset/puf_splitted_dataset/val'
TEST_DIR = '/data/shurong/puf/puf_dataset/puf_splitted_dataset/test'
RESULT_DIR = './effnet_v2l_pretrained_evaluation_results'
WRONG_IMG_DIR = os.path.join(RESULT_DIR, 'wrong_cases')
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(WRONG_IMG_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def load_model(device):
    model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()  # Output a 1280-d pooled feature.
    return model.to(device).eval()

@torch.no_grad()
def extract_features(data_dir, model):
    feats, labels, paths = [], [], []
    files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png'))])
    for f in tqdm(files, desc=f'Extract {os.path.basename(data_dir)}'):
        path = os.path.join(data_dir, f)
        img = ImageOps.autocontrast(Image.open(path).convert('RGB'))
        x = transform(img).unsqueeze(0).to(device)
        feat = torch.nn.functional.normalize(model(x), p=2, dim=1).cpu().numpy()
        feats.append(feat)
        labels.append(f.split('_')[0])
        paths.append(path)
    return np.vstack(feats), np.array(labels), paths

def get_similarity(feats, labels, paths):
    g = np.array(['_orig' in os.path.basename(p) for p in paths])
    q = np.array(['_aug' in os.path.basename(p) for p in paths])
    gf, gl = feats[g], labels[g]
    qf, ql = feats[q], labels[q]
    gp = [paths[i] for i, x in enumerate(g) if x]
    qp = [paths[i] for i, x in enumerate(q) if x]
    sim = qf @ gf.T
    return sim, gl, ql, gp, qp

def find_threshold(feats, labels, paths):
    sim, gl, ql, _, _ = get_similarity(feats, labels, paths)
    same = ql[:, None] == gl[None, :]
    pos, neg = sim[same], sim[~same]
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    scores = np.concatenate([pos, neg])
    fpr, tpr, thresholds = roc_curve(y, scores)
    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    threshold = thresholds[eer_idx]
    eer = (fpr[eer_idx] + 1 - tpr[eer_idx]) / 2
    val_auc = auc(fpr, tpr)

    print('\n' + '=' * 60)
    print('Validation')
    print(f'PUFs: {len(gl)}, Queries: {len(ql)}')
    print(f'AUC: {val_auc:.6f}')
    print(f'EER: {eer:.6f}')
    print(f'Threshold: {threshold:.6f}')
    print('=' * 60)

    with open(os.path.join(RESULT_DIR, 'pretrained_validation_roc.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['threshold', 'FPR', 'TPR', 'FNR'])
        for th, fp, tp in zip(thresholds, fpr, tpr):
            writer.writerow([th, fp, tp, 1 - tp])

    return threshold

def evaluate_test(feats, labels, paths, threshold):
    sim, gl, ql, gp, qp = get_similarity(feats, labels, paths)

    best_idx = np.argmax(sim, axis=1)
    pred = gl[best_idx]
    top1 = np.mean(pred == ql)

    same = ql[:, None] == gl[None, :]
    pos, neg = sim[same], sim[~same]

    tar = np.mean(pos >= threshold)
    far = np.mean(neg >= threshold)
    frr = 1 - tar

    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    scores = np.concatenate([pos, neg])
    fpr, tpr, _ = roc_curve(y, scores)
    auc_score = auc(fpr, tpr)

    print('\n' + '=' * 70)
    print(f'{"Model":<25}{"Top1":>10}{"AUC":>10}{"TAR":>10}{"FAR":>10}{"FRR":>10}')
    print('-' * 70)
    print(f'{"Pretrained EffNet-V2L":<25}{top1:>10.2%}{auc_score:>10.4f}{tar:>10.2%}{far:>10.4%}{frr:>10.4%}')
    print('-' * 70)
    print(f'Validation threshold: {threshold:.6f}')
    print('=' * 70)

    wrong_cases = []
    for i in range(len(ql)):
        if pred[i] != ql[i]:
            idx = best_idx[i]
            case = {
                'query_filename': os.path.basename(qp[i]),
                'true_id': ql[i],
                'pred_id': pred[i],
                'similarity': f'{sim[i, idx]:.4f}',
                'gallery_match': os.path.basename(gp[idx])
            }
            wrong_cases.append(case)
            save_name = f'ERR_True_{ql[i]}_Pred_{pred[i]}_{os.path.basename(qp[i])}'
            shutil.copy(qp[i], os.path.join(WRONG_IMG_DIR, save_name))

    csv_path = os.path.join(RESULT_DIR, 'test_wrong_cases_report.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['query_filename', 'true_id', 'pred_id', 'similarity', 'gallery_match'])
        writer.writeheader()
        writer.writerows(wrong_cases)

    summary_path = os.path.join(RESULT_DIR, 'test_results.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Top1', 'AUC', 'TAR', 'FAR', 'FRR', 'Threshold'])
        writer.writerow(['Pretrained_EffNet_V2L', top1, auc_score, tar, far, frr, threshold])

    print(f'Wrong cases: {len(wrong_cases)}')
    print(f'Results saved to: {summary_path}')

def main():
    model = load_model(device)

    print('\n[1] Validation feature extraction')
    val_feats, val_labels, val_paths = extract_features(VAL_DIR, model)

    print('\n[2] Find threshold')
    threshold = find_threshold(val_feats, val_labels, val_paths)

    print('\n[3] Test feature extraction')
    test_feats, test_labels, test_paths = extract_features(TEST_DIR, model)

    print('\n[4] Test evaluation')
    evaluate_test(test_feats, test_labels, test_paths, threshold)

if __name__ == '__main__':
    main()