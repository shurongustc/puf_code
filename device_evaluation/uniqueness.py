import os,sys
sys.path.insert(0,"/data/shurong/puf/facebookresearch_dinov2_main")
import torch,numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from dinov2.models import vision_transformer as vits

TEST_DIR="/data/shurong/puf/puf_dataset/puf_splitted_dataset/test"
MODEL_PATH="/data/shurong/puf/checkpoints/dinov2_puf_best.pth"
CKPT_DINO=MODEL_PATH
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

transform=transforms.Compose([transforms.Resize((518,518)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

model=vits.__dict__["vit_base"](patch_size=14,img_size=518,init_values=1.0)
ckpt=torch.load(CKPT_DINO,map_location="cpu",weights_only=False)
sd=ckpt["model"] if "model" in ckpt else ckpt
sd={k.replace("module.",""):v for k,v in sd.items()}
model.load_state_dict(sd,strict=False)
model.to(DEVICE).eval()

def feature(path):
    x=transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x=model(x)
        x=torch.nn.functional.normalize(x,dim=1)
    return x.cpu().numpy()[0]

puf={}
for f in sorted(os.listdir(TEST_DIR)):
    if f.lower().endswith((".jpg",".png",".jpeg")):
        pid=f.split("_")[0]
        puf.setdefault(pid,[]).append(os.path.join(TEST_DIR,f))

puf={k:np.stack([feature(x) for x in v]) for k,v in puf.items() if len(v)>1}
print("PUF:",len(puf))

intra=[]
for x in tqdm(puf.values()):
    s=cosine_similarity(x)
    n=len(x)
    intra.extend([s[i,j] for i in range(n) for j in range(i+1,n)])
intra=np.array(intra)

inter=[]
ids=list(puf)
for i in tqdm(range(len(ids))):
    for j in range(i+1,len(ids)):
        inter.extend(cosine_similarity(puf[ids[i]],puf[ids[j]]).ravel())
inter=np.array(inter)


mu_intra, std_intra = intra.mean(), intra.std()
mu_inter, std_inter = inter.mean(), inter.std()

#  Decidability Index (d-prime)
d_prime = np.abs(mu_intra - mu_inter) / np.sqrt(0.5 * (std_intra**2 + std_inter**2))

# Fisher Discriminant Ratio (FDR)
fdr = (mu_intra - mu_inter)**2 / (std_intra**2 + std_inter**2)

print(f"Intra - Mean: {mu_intra:.4f}, Std: {std_intra:.4f}")
print(f"Inter - Mean: {mu_inter:.4f}, Std: {std_inter:.4f}")
print(f"Decidability Index (d'): {d_prime:.4f}")
print(f"Fisher Discriminant Ratio (FDR): {fdr:.4f}")

plt.figure(figsize=(7,4))
sns.histplot(inter,bins=100,stat="density",alpha=.5,label="Inter-PUF")
sns.histplot(intra,bins=100,stat="density",alpha=.5,label="Intra-PUF")
plt.xlabel("Cosine similarity")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("puf_uniqueness_distribution.png",dpi=300)
plt.show()