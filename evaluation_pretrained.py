import torch,os,sys
import numpy as np
from PIL import Image,ImageOps
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import roc_curve,auc

@torch.no_grad()
def get_features(data_dir,model,transform,device):
    model.eval()
    files=sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg','.png'))])
    orig_f,aug_f,orig_l,aug_l=[],[],[],[]
    print(f"📦 Extracting features from {os.path.basename(data_dir)}...")
    for f in tqdm(files):
        img=ImageOps.autocontrast(Image.open(os.path.join(data_dir,f)).convert('RGB'))
        out=model(transform(img).unsqueeze(0).to(device))
        feat=out["x_norm_clstoken"] if isinstance(out,dict) else out
        feat=torch.nn.functional.normalize(feat,p=2,dim=1).cpu().numpy()
        label=f.split('_')[0]
        if '_orig' in f:
            orig_f.append(feat);orig_l.append(label)
        elif '_aug' in f:
            aug_f.append(feat);aug_l.append(label)
    return np.vstack(orig_f),orig_l,np.vstack(aug_f),aug_l

def search_eer_threshold(gal_f,gal_l,que_f,que_l):
    print("🎯 Searching for the EER threshold...")
    sims=que_f@gal_f.T
    pos,neg=[],[]
    for i in range(len(que_f)):
        for j in range(len(gal_f)):
            (pos if que_l[i]==gal_l[j] else neg).append(sims[i,j])
    scores=np.concatenate([pos,neg])
    labels=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
    fpr,tpr,th=roc_curve(labels,scores)
    fnr=1-tpr
    idx=np.argmin(abs(fpr-fnr))
    return th[idx],fpr[idx]

def evaluate_pretrained():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    REPO_PATH='/data/shurong/puf/facebookresearch_dinov2_main'
    CKPT='/data/shurong/puf/checkpoints/dino_vitb14_pretrain.pth'
    VAL_DIR='/data/shurong/puf/puf_dataset/puf_splitted_dataset/val'
    TEST_DIR='/data/shurong/puf/puf_dataset/puf_splitted_dataset/test'
    sys.path.append(REPO_PATH) if REPO_PATH not in sys.path else None
    from dinov2.models import vision_transformer as vits

    print(f"🚀 Loading model: {CKPT}")
    model=vits.__dict__['vit_base'](patch_size=14,img_size=518,init_values=1.0,block_chunks=0)

    ckpt=torch.load(CKPT,map_location='cpu')
    sd=ckpt['model'] if 'model' in ckpt else ckpt
    sd={k.replace('module.',''):v for k,v in sd.items()}
    msg=model.load_state_dict(sd,strict=False)
    print(f"Missing:{len(msg.missing_keys)}, Unexpected:{len(msg.unexpected_keys)}")

    model.to(device).eval()

    transform=transforms.Compose([
        transforms.Resize((518,518)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

    v_gal_f,v_gal_l,v_que_f,v_que_l=get_features(VAL_DIR,model,transform,device)
    val_th,val_eer=search_eer_threshold(v_gal_f,v_gal_l,v_que_f,v_que_l)
    print(f"Val threshold:{val_th:.4f}")
    print(f"Val EER:{val_eer:.2%}")

    t_gal_f,t_gal_l,t_que_f,t_que_l=get_features(TEST_DIR,model,transform,device)
    sims=t_que_f@t_gal_f.T

    pred=np.argmax(sims,axis=1)
    top1=np.mean(np.array(t_gal_l)[pred]==np.array(t_que_l))

    mask=np.array(t_que_l)[:,None]==np.array(t_gal_l)[None,:]
    pos=sims[mask]
    neg=sims[~mask]

    tar=np.mean(pos>=val_th)
    far=np.mean(neg>=val_th)
    frr=1-tar

    y=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
    s=np.concatenate([pos,neg])
    fpr,tpr,_=roc_curve(y,s)
    roc_auc=auc(fpr,tpr)

    print("\n"+"-"*60)
    print(f"{'Model':<15}{'Top-1':>12}{'AUC':>12}{'TAR':>12}{'FAR':>12}{'FRR':>12}")
    print("-"*60)
    print(f"{'DINOv2':<15}{top1:>12.2%}{roc_auc:>12.4f}{tar:>12.2%}{far:>12.4%}{frr:>12.2%}")
    print("-"*60)

if __name__=="__main__":
    evaluate_pretrained()