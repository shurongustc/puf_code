import torch,os,sys
import numpy as np
from PIL import Image,ImageOps
import torchvision.transforms as T

REPO='./facebookresearch_dinov2_main'
CKPT='./checkpoints/dinov2_puf_best.pth'
GALLERY='./puf_dataset/puf_splitted_dataset/test'
GALLERY_FEAT='./gallery_feats.npy'
GALLERY_NAME='./gallery_names.npy'
THRESHOLD=0.5436
TOPK=5

sys.path.append(REPO)
from dinov2.models import vision_transformer as vits

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PUFMatcher:
    def __init__(self):
        self.model=vits.vit_base(patch_size=14,img_size=518)
        ck=torch.load(CKPT,map_location='cpu',weights_only=False)
        ck=ck['model'] if 'model' in ck else ck
        ck={k.replace('module.',''):v for k,v in ck.items()}
        self.model.load_state_dict(ck,strict=False)
        self.model.to(device).eval()
        self.tf=T.Compose([T.Resize((518,518)),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
        self.names=[]
        self.feats=None
        self.build_gallery()

    @torch.inference_mode()
    def encode(self,img):
        if isinstance(img,str): img=Image.open(img).convert('RGB')
        else: img=img.convert('RGB')
        img=ImageOps.autocontrast(img)
        x=self.tf(img).unsqueeze(0).to(device)
        f=self.model(x)
        if isinstance(f,dict): f=f['x_norm_clstoken']
        return torch.nn.functional.normalize(f,p=2,dim=1).cpu().numpy()

    def build_gallery(self):
        if os.path.exists(GALLERY_FEAT) and os.path.exists(GALLERY_NAME):
            print('Loading gallery npy...')
            self.feats=np.load(GALLERY_FEAT)
            self.names=np.load(GALLERY_NAME,allow_pickle=True).tolist()
            print(f'Gallery loaded: {len(self.names)}')
            return
        print('Building gallery...')
        feats=[]
        names=[]
        for f in sorted(os.listdir(GALLERY)):
            if '_orig' in f:
                feats.append(self.encode(os.path.join(GALLERY,f))[0])
                names.append(f.replace('_orig',''))
        self.feats=np.asarray(feats,dtype=np.float32)
        self.names=names
        np.save(GALLERY_FEAT,self.feats)
        np.save(GALLERY_NAME,np.asarray(self.names))
        print(f'Gallery saved: {len(self.names)}')

    def match_topk(self,img,k=TOPK):
        q=self.encode(img)
        sim=(q@self.feats.T)[0]
        idx=np.argsort(sim)[::-1][:k]
        topk=[(self.names[i],float(sim[i])) for i in idx]
        score=topk[0][1]
        return {'result':'PASS' if score>=THRESHOLD else 'REJECT','similarity':round(score,4),'matched_id':topk[0][0],'topk':topk}

    def match(self,img):
        return self.match_topk(img,1)

if __name__=='__main__':
    matcher=PUFMatcher()
    print(matcher.match_topk('./puf_dataset/puf_splitted_dataset/test/20251230145930368_aug_0.jpg'))