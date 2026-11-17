import torch,os,sys,csv,numpy as np,cv2,matplotlib.pyplot as plt
from PIL import Image,ImageOps
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_l
from tqdm import tqdm
from sklearn.metrics import roc_curve,auc

# === 1. Paths & Global Configuration ===
VAL_DIR='/data/shurong/puf/puf_dataset/puf_splitted_dataset/val'
TEST_DIR='/data/shurong/puf/puf_dataset/puf_splitted_dataset/test'
REPO_PATH_DINO='/data/shurong/puf/facebookresearch_dinov2_main'
CKPT_DINO='./checkpoints/dinov2_puf_best_50epochs.pth'
CKPT_EFFNET='./checkpoints/effnet_v2l_best.pth'
RESULT_DIR='./benchmarking_results_val_threshold'
os.makedirs(RESULT_DIR,exist_ok=True)
if REPO_PATH_DINO not in sys.path:sys.path.append(REPO_PATH_DINO)
from dinov2.models import vision_transformer as vits

# === 2. Deep Learning Model Metrics Calculation (Using the Threshold Saved in Checkpoint) ===
@torch.no_grad()
def get_dl_model_metrics(model, transform, device, data_dir, threshold):
    files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png'))])
    feats, labels = [], []
    for f in tqdm(files, desc="Extracting features", leave=False):
        img = ImageOps.autocontrast(Image.open(os.path.join(data_dir, f)).convert('RGB'))
        img_t = transform(img).unsqueeze(0).to(device)
        feat = torch.nn.functional.normalize(model(img_t), p=2, dim=1).cpu().numpy()
        feats.append(feat); labels.append(f.split('_')[0])
    feats = np.vstack(feats); labels = np.array(labels)
    g_idx = np.array(['_orig' in f for f in files]); q_idx = np.array(['_aug' in f for f in files])
    g_feats, g_labels = feats[g_idx], labels[g_idx]
    q_feats, q_labels = feats[q_idx], labels[q_idx]
    sim_matrix = q_feats @ g_feats.T
    mask = q_labels[:, np.newaxis] == g_labels[np.newaxis, :]
    pos_scores = sim_matrix[mask]; neg_scores = sim_matrix[~mask]
    top1_acc = np.sum(g_labels[np.argmax(sim_matrix, axis=1)] == q_labels) / len(q_labels)
    return pos_scores, neg_scores, top1_acc, threshold

# === 3. SIFT + FLANN Evaluation (Using Validation Set to Compute Threshold via Youden's J Statistic) ===
def get_sift_metrics(val_dir, test_dir):
    sift = cv2.SIFT_create(nfeatures=500)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    def extract_features(data_dir):
        files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png'))])
        g_files = [f for f in files if '_orig' in f]
        q_files = [f for f in files if '_aug' in f]
        
        def _extract(flist, desc_msg):
            features = {}
            for f in tqdm(flist, desc=desc_msg, leave=False):
                img = cv2.imread(os.path.join(data_dir, f), cv2.IMREAD_GRAYSCALE)
                if img is None: features[f] = None; continue
                img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
                img = cv2.equalizeHist(img)
                _, des = sift.detectAndCompute(img, None)
                features[f] = des
            return features, g_files if '_orig' in flist[0] else q_files
            
        g_feat = _extract(g_files, "Extracting SIFT features (Gallery)")
        q_feat = _extract(q_files, "Extracting SIFT features (Query)")
        return g_files, q_files, g_feat[0], q_feat[0]

    # Compute optimal threshold on Validation Set
    print("--- Computing SIFT Optimal Threshold on Validation Set ---")
    val_g_files, val_q_files, val_g_features, val_q_features = extract_features(val_dir)
    val_pos, val_neg = [], []
    for q_f in tqdm(val_q_files, desc="SIFT Val Matching"):
        q_des = val_q_features[q_f]; q_label = q_f.split('_')[0]
        for g_f in val_g_files:
            g_des = val_g_features[g_f]; g_label = g_f.split('_')[0]
            if q_des is not None and g_des is not None:
                try:
                    matches = flann.knnMatch(q_des, g_des, k=2)
                    score = sum(1 for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance) / len(q_des)
                except cv2.error: score = 0
            else: score = 0
            if q_label == g_label: val_pos.append(score)
            else: val_neg.append(score)
            
    val_pos_arr, val_neg_arr = np.array(val_pos), np.array(val_neg)
    val_y_true = np.concatenate([np.ones_like(val_pos_arr), np.zeros_like(val_neg_arr)])
    val_y_score = np.concatenate([val_pos_arr, val_neg_arr])
    fpr_v, tpr_v, thresholds_v = roc_curve(val_y_true, val_y_score)
    sift_threshold = thresholds_v[np.argmax(tpr_v - fpr_v)]
    print(f"✅ SIFT Optimal Threshold on Validation Set: {sift_threshold:.4f}")

    # Evaluate on Test Set using the validation-derived threshold
    print("--- Evaluating SIFT on Test Set ---")
    test_g_files, test_q_files, test_g_features, test_q_features = extract_features(test_dir)
    pos_scores, neg_scores = [], []; hits = 0
    for q_f in tqdm(test_q_files, desc="SIFT Test Matching"):
        q_des = test_q_features[q_f]; q_label = q_f.split('_')[0]
        max_score, best_label = -1, None
        for g_f in test_g_files:
            g_des = test_g_features[g_f]; g_label = g_f.split('_')[0]
            if q_des is not None and g_des is not None:
                try:
                    matches = flann.knnMatch(q_des, g_des, k=2)
                    score = sum(1 for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance) / len(q_des)
                except cv2.error: score = 0
            else: score = 0
            if q_label == g_label: pos_scores.append(score)
            else: neg_scores.append(score)
            if score > max_score: max_score, best_label = score, g_label
        if best_label == q_label: hits += 1
        
    return np.array(pos_scores), np.array(neg_scores), hits / len(test_q_files), sift_threshold

# === 4. Run Full Benchmark Workflow ===
def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    # --- 4.1 DINOv2 Model Preparation ---
    print("\n--- Loading and Evaluating DINOv2 ---")
    m_dino = vits.__dict__['vit_base'](patch_size=14, img_size=518, init_values=1.0)
    ckpt_d = torch.load(CKPT_DINO, map_location='cpu', weights_only=False)
    sd_d = ckpt_d['model'] if 'model' in ckpt_d else ckpt_d
    m_dino.load_state_dict({k.replace('module.', ''): v for k, v in sd_d.items()}, strict=False)
    
    model_dino_full = torch.nn.Sequential(
        m_dino,
        torch.nn.LayerNorm(768)
    ).to(device)
    model_dino_full.eval()
    
    t_dino = transforms.Compose([transforms.Resize((518, 518)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    dino_th = ckpt_d.get('threshold', 0.4809) if isinstance(ckpt_d, dict) else 0.4809
    results['PUF-DINO'] = get_dl_model_metrics(model_dino_full, t_dino, device, TEST_DIR, threshold=dino_th)

    # --- 4.2 EfficientNet-V2-L Model Preparation ---
    print("\n--- Loading and Evaluating EfficientNet-V2-L ---")
    m_eff = efficientnet_v2_l(weights=None)
    in_feat = m_eff.classifier[1].in_features
    m_eff.classifier = torch.nn.Sequential(torch.nn.Linear(in_feat, 768), torch.nn.LayerNorm(768), torch.nn.Dropout(p=0.4))
    ckpt_e = torch.load(CKPT_EFFNET, map_location='cpu', weights_only=False)
    sd_e = ckpt_e['model'] if 'model' in ckpt_e else ckpt_e
    m_eff.load_state_dict({k.replace('module.', ''): v for k, v in sd_e.items()})
    m_eff.to(device).eval()
    
    t_eff = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    eff_th = ckpt_e.get('threshold', 0.4956) if isinstance(ckpt_e, dict) else 0.4950
    results['PUF-EffNet-V2L'] = get_dl_model_metrics(m_eff, t_eff, device, TEST_DIR, threshold=eff_th)

    # --- 4.3 SIFT Model Preparation ---
    print("\n--- Evaluating SIFT + FLANN ---")
    results['SIFT'] = get_sift_metrics(VAL_DIR, TEST_DIR)

    # --- 5. Performance Report & CSV Export ---
    print("\n" + "*" * 35)
    print(f"{'Model':<15} | {'Top-1 Acc':<10} | {'Threshold':<10} | {'AUC':<10} | {'TAR (at Th)':<12} | {'FAR (at Th)':<12}")
    print("-" * 105)

    csv_path = os.path.join(RESULT_DIR, 'benchmark_results_val_threshold.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Top-1 Accuracy', 'Threshold', 'AUC', 'TAR@threshold', 'FAR@threshold', 'FRR@threshold'])

        plt.figure(figsize=(10, 8))

        for name, (pos, neg, acc, th) in results.items():
            y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
            y_score = np.concatenate([pos, neg])
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            tar_th = np.mean(pos >= th)
            far_th = np.mean(neg >= th)
            frr_th = 1.0 - tar_th

            print(f"{name:<15} | {acc:<10.2%} | {th:<10.4f} | {roc_auc:<10.4f} | {tar_th:<12.2%} | {far_th:<12.2%}")

            writer.writerow([
                name,
                f'{acc:.6f}',
                f'{th:.6f}',
                f'{roc_auc:.6f}',
                f'{tar_th:.6f}',
                f'{far_th:.6f}',
                f'{frr_th:.6f}'
            ])

            plt.plot(fpr, tpr, lw=2.5, label=f'{name} (AUC={roc_auc:.4f})')
            idx = np.argmin(np.abs(thresholds - th))
            plt.scatter(fpr[idx], tpr[idx], s=60, zorder=5)

    print(f"\nCSV results saved to: {csv_path}")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.2)
    plt.xscale('log'); plt.xlim([1e-4, 1.0]); plt.ylim([0.0, 1.05])
    
    plt.xlabel('FAR', fontname='Arial', fontsize=18, fontweight='bold')
    plt.ylabel('TAR', fontname='Arial', fontsize=18, fontweight='bold')
    
    plt.xticks(fontname='Arial', fontsize=18, fontweight='bold')
    plt.yticks(fontname='Arial', fontsize=18, fontweight='bold')
    
    legend = plt.legend(loc="lower right")
    for text in legend.get_texts():
        text.set_fontname('Arial')
        text.set_fontsize(18)
        text.set_fontweight('bold')
        
    plt.grid(which='both', alpha=0.3)
    plt.savefig(os.path.join(RESULT_DIR, 'final_comparison_val_threshold.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_benchmark()