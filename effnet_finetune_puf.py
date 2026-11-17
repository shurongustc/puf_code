import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter 
import torch.multiprocessing as mp
import os
import numpy as np
import csv
import random
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_curve

# === 1. Dataset class ===
class PUFAnchorDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        files = sorted([f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.png'))])
        id_groups = defaultdict(lambda: {'orig': None, 'augs': []})
        for f in files:
            pid = f.split('_')[0]
            if '_orig' in f: id_groups[pid]['orig'] = f
            elif '_aug' in f: id_groups[pid]['augs'].append(f)
        self.pairs = []
        for pid, content in id_groups.items():
            if content['orig'] and content['augs']:
                for aug_file in content['augs']:
                    self.pairs.append((pid, content['orig'], aug_file))
        self.pids = list(id_groups.keys())
        self.all_files = files

    def __len__(self): return len(self.pairs)

    def load_img(self, f):
        img = ImageOps.autocontrast(Image.open(os.path.join(self.data_path, f)).convert('RGB'))
        return img

    def __getitem__(self, idx):
        pid, a_file, p_file = self.pairs[idx]
        while True:
            neg_file = random.choice(self.all_files)
            if neg_file.split('_')[0] != pid: break
        a, p, n = self.load_img(a_file), self.load_img(p_file), self.load_img(neg_file)
        if self.transform: a, p, n = self.transform(a), self.transform(p), self.transform(n)
        return a, p, n

# === 2. Validation function ===
@torch.no_grad()
def validate_and_find_threshold(model, val_dir, transform, device):
    model.eval()
    files = sorted([f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.png'))])
    orig_features, aug_features = [], []
    orig_labels, aug_labels = [], []
    
    for f in files:
        img = ImageOps.autocontrast(Image.open(os.path.join(val_dir, f)).convert('RGB'))
        img_tensor = transform(img).unsqueeze(0).to(device)
        feat = torch.nn.functional.normalize(model(img_tensor), p=2, dim=1).cpu().numpy()
        label = f.split('_')[0]
        if '_orig' in f:
            orig_features.append(feat); orig_labels.append(label)
        else:
            aug_features.append(feat); aug_labels.append(label)
            
    if not orig_features or not aug_features: return 0.0, 0.0
    gallery, queries = np.vstack(orig_features), np.vstack(aug_features)
    sim_matrix = queries @ gallery.T
    
    correct = 0
    for i in range(len(queries)):
        if aug_labels[i] == orig_labels[np.argmax(sim_matrix[i])]: correct += 1
    top1_acc = correct / len(queries)
    
    pos_scores, neg_scores = [], []
    for i in range(len(queries)):
        for j in range(len(gallery)):
            score = sim_matrix[i, j]
            if aug_labels[i] == orig_labels[j]: pos_scores.append(score)
            else: neg_scores.append(score)
            
    scores = np.array(pos_scores + neg_scores)
    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer_idx = np.nanargmin(np.absolute((fpr - (1 - tpr))))
    return top1_acc, thresholds[eer_idx]

# === 3. Model construction (fix in-place conflicts) ===
def get_efficientnet_v2l_model(out_dim=768):
    weights = EfficientNet_V2_L_Weights.DEFAULT
    model = efficientnet_v2_l(weights=weights)
    
    # Core fix 1: Recursively disable all in-place operations within the model.
    for m in model.modules():
        if hasattr(m, 'inplace'):
            m.inplace = False
    
    in_features = model.classifier[1].in_features
    # Core fix 2: Use LayerNorm instead of BatchNorm to avoid multi-GPU synchronization conflicts; disable Dropout.
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features, out_dim),
        torch.nn.LayerNorm(out_dim),
        torch.nn.Dropout(p=0.4, inplace=False)
    )
    
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.features[6:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    return model

# === 4. Training process ===
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29515' 
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def train_worker(rank, world_size, train_dir, val_dir):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    writer = None
    csv_path = "effnet_v2l_puf_384_results_50epochs.csv"
    if rank == 0:
        writer = SummaryWriter(log_dir="runs/effnet_v2l_384_exp")
        with open(csv_path, mode='w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_acc', 'val_threshold'])

    model = get_efficientnet_v2l_model(out_dim=768).to(device)
    model = DDP(model, device_ids=[rank])

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    train_ds = PUFAnchorDataset(train_dir, transform=transform)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    # Merging batches increases GPU memory usage; reduce batch_size if an OOM occurs.
    train_loader = DataLoader(train_ds, batch_size=8, sampler=train_sampler, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = torch.nn.TripletMarginLoss(margin=0.4, p=2).to(device)
    scaler = GradScaler('cuda')

    best_acc = 0.0
    for epoch in range(50):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss_sum = 0.0
        
        pbar = tqdm(train_loader, desc=f"Rank {rank} | Epoch {epoch}") if rank == 0 else train_loader
        for i, (a, p, n) in enumerate(pbar):
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                # Core fix 3: Merge the forward pass to avoid version conflicts.
                combined = torch.cat([a, p, n], dim=0)
                combined_out = model(combined)
                
                bs = a.size(0)
                af = torch.nn.functional.normalize(combined_out[:bs], p=2, dim=1)
                pf = torch.nn.functional.normalize(combined_out[bs:2*bs], p=2, dim=1)
                nf = torch.nn.functional.normalize(combined_out[2*bs:], p=2, dim=1)
                
                loss = criterion(af, pf, nf)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss_sum += loss.item()
            
            if rank == 0:
                writer.add_scalar('Loss/train_step', loss.item(), epoch * len(train_loader) + i)

        if rank == 0:
            avg_loss = epoch_loss_sum / len(train_loader)
            acc, threshold = validate_and_find_threshold(model.module, val_dir, transform, device)
            
            writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
            writer.add_scalar('Accuracy/val_top1', acc, epoch)
            writer.add_scalar('Metrics/threshold_eer', threshold, epoch)
            
            with open(csv_path, mode='a', newline='') as f:
                csv.writer(f).writerow([epoch, f"{avg_loss:.6f}", f"{acc:.4f}", f"{threshold:.4f}"])
            
            print(f"📊 [EffNet-V2L] Epoch {epoch}: Loss {avg_loss:.4f}, Acc {acc:.2%}, Thr {threshold:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                torch.save({'model': model.module.state_dict(), 'acc': acc}, "effnet_v2l_best.pth")

    if rank == 0: writer.close()
    dist.destroy_process_group()

if __name__ == "__main__":
    TRAIN_DIR = '/data/shurong/puf/puf_dataset/puf_splitted_dataset/train'
    VAL_DIR   = '/data/shurong/puf/puf_dataset/puf_splitted_dataset/val'
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(train_worker, args=(WORLD_SIZE, TRAIN_DIR, VAL_DIR), nprocs=WORLD_SIZE, join=True)