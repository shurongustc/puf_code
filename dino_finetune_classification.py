import os,sys,csv
import numpy as np
from PIL import Image,ImageOps
from tqdm import tqdm
from sklearn.metrics import roc_curve
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast,GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

class PUFClassificationDataset(Dataset):
    def __init__(self,data_path,transform=None):
        self.data_path=data_path
        self.transform=transform
        files=sorted([f for f in os.listdir(data_path) if f.lower().endswith(('.jpg','.png'))])
        self.pid_to_label={}
        self.samples=[]
        for f in files:
            pid=f.split('_')[0]
            if pid not in self.pid_to_label:self.pid_to_label[pid]=len(self.pid_to_label)
            self.samples.append((f,self.pid_to_label[pid]))
        self.num_classes=len(self.pid_to_label)
    def __len__(self):return len(self.samples)
    def __getitem__(self,idx):
        f,label=self.samples[idx]
        img=ImageOps.autocontrast(Image.open(os.path.join(self.data_path,f)).convert('RGB'))
        if self.transform:img=self.transform(img)
        return img,label

@torch.no_grad()
def validate_and_find_threshold(model,val_dir,transform,device):
    model.eval()
    backbone=model.module[0] if hasattr(model,'module') else model[0]
    files=sorted([f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg','.png'))])
    orig_f,aug_f,orig_l,aug_l=[],[],[],[]
    for f in files:
        img=ImageOps.autocontrast(Image.open(os.path.join(val_dir,f)).convert('RGB'))
        feat=F.normalize(backbone(transform(img).unsqueeze(0).to(device)),p=2,dim=1).cpu().numpy()
        pid=f.split('_')[0]
        if '_orig' in f:orig_f.append(feat);orig_l.append(pid)
        else:aug_f.append(feat);aug_l.append(pid)
    if not orig_f or not aug_f:return 0.,0.
    gallery,query=np.vstack(orig_f),np.vstack(aug_f)
    sim=query@gallery.T
    acc=sum(aug_l[i]==orig_l[np.argmax(sim[i])] for i in range(len(query)))/len(query)
    pos,neg=[],[]
    for i in range(len(query)):
        for j in range(len(gallery)):
            (pos if aug_l[i]==orig_l[j] else neg).append(sim[i,j])
    scores=np.array(pos+neg)
    labels=np.array([1]*len(pos)+[0]*len(neg))
    fpr,tpr,thr=roc_curve(labels,scores)
    eer=np.nanargmin(np.abs(fpr-(1-tpr)))
    return acc,thr[eer]

def setup(rank,world_size):
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='29527'
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',rank=rank,world_size=world_size)

def train_worker(rank,world_size,train_dir,val_dir,ckpt_path,repo_dir):
    setup(rank,world_size)
    device=torch.device(f'cuda:{rank}')
    writer=None
    csv_path='puf_training_log.csv'
    if rank==0:
        writer=SummaryWriter('runs/puf_dino_cls')
        with open(csv_path,'w',newline='') as f:csv.writer(f).writerow(['epoch','loss','acc','thr'])
    if repo_dir not in sys.path:sys.path.append(repo_dir)
    from dinov2.models import vision_transformer as vits
    backbone=vits.__dict__['vit_base'](patch_size=14,img_size=518,init_values=1.0)
    sd=torch.load(ckpt_path,map_location='cpu')
    if 'model' in sd:sd=sd['model']
    new_sd={f"blocks.0.{'.'.join(k.split('.')[1:])}" if k.startswith('blocks.') and k.split('.')[1].isdigit() else k:v for k,v in sd.items()}
    backbone.load_state_dict(new_sd,strict=False)
    for p in backbone.parameters():p.requires_grad=False
    for p in backbone.blocks[-4:].parameters():p.requires_grad=True
    for p in backbone.norm.parameters():p.requires_grad=True
    transform=transforms.Compose([transforms.Resize((518,518)),transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    train_ds=PUFClassificationDataset(train_dir,transform)
    model=nn.Sequential(backbone,nn.LayerNorm(768),nn.Dropout(0.4),nn.Linear(768,train_ds.num_classes)).to(device)
    model=DDP(model,device_ids=[rank])
    sampler=DistributedSampler(train_ds,num_replicas=world_size,rank=rank,shuffle=True)
    loader=DataLoader(train_ds,batch_size=12,sampler=sampler,num_workers=8,pin_memory=True)
    optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=2e-6)
    criterion=nn.CrossEntropyLoss().to(device)
    scaler=GradScaler()
    best_acc=0.
    for epoch in range(50):
        sampler.set_epoch(epoch)
        model.train()
        loss_sum=0.
        pbar=tqdm(loader,desc=f'Epoch {epoch}') if rank==0 else loader
        for step,(imgs,labels) in enumerate(pbar):
            imgs,labels=imgs.to(device,non_blocking=True),labels.to(device)
            optimizer.zero_grad()
            with autocast():
                loss=criterion(model(imgs),labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum+=loss.item()
            if rank==0:writer.add_scalar('Loss/train_step',loss.item(),epoch*len(loader)+step)
        if rank==0:
            avg_loss=loss_sum/len(loader)
            acc,thr=validate_and_find_threshold(model,val_dir,transform,device)
            writer.add_scalar('Loss/train_epoch',avg_loss,epoch)
            writer.add_scalar('Accuracy/val_top1',acc,epoch)
            writer.add_scalar('Threshold/eer',thr,epoch)
            with open(csv_path,'a',newline='') as f:csv.writer(f).writerow([epoch,f'{avg_loss:.6f}',f'{acc:.6f}',f'{thr:.6f}'])
            print(f'Epoch {epoch}:Loss {avg_loss:.4f},Acc {acc:.2%},Thr {thr:.4f}')
            if acc>best_acc:
                best_acc=acc
                torch.save({'model':model.module.state_dict(),'backbone':model.module[0].state_dict(),'classifier':model.module[3].state_dict(),'threshold':thr,'num_classes':train_ds.num_classes},'dinov2_puf_best.pth')
    if rank==0:writer.close()
    dist.destroy_process_group()

if __name__=='__main__':
    WORLD_SIZE=2
    TRAIN_DIR='/data/shurong/puf/puf_dataset/puf_splitted_dataset/train'
    VAL_DIR='/data/shurong/puf/puf_dataset/puf_splitted_dataset/val'
    CKPT_PATH='/data/shurong/puf/checkpoints/dino_vitb14_pretrain.pth'
    REPO_PATH='/data/shurong/puf/facebookresearch_dinov2_main'
    mp.spawn(train_worker,args=(WORLD_SIZE,TRAIN_DIR,VAL_DIR,CKPT_PATH,REPO_PATH),nprocs=WORLD_SIZE,join=True)