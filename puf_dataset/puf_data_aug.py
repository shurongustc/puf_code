import cv2, numpy as np, matplotlib.pyplot as plt
import albumentations as A

IMG = "./puf_dataset_gray_original/20260105151343592.jpg"
OUT = "./Supplementary_Fig1_perturbations.png"

img = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB)

# ---------- fixed augmentations ----------
def aug(op): return op(image=img)["image"]

rows = [
("Rotation",[
    aug(A.SafeRotate(limit=(30,30),border_mode=cv2.BORDER_REFLECT_101,p=1)),
    aug(A.SafeRotate(limit=(90,90),border_mode=cv2.BORDER_REFLECT_101,p=1)),
    aug(A.SafeRotate(limit=(180,180),border_mode=cv2.BORDER_REFLECT_101,p=1))
]),

("Perspective",[
    aug(A.Perspective(scale=(0.05,0.05),p=1)),
    aug(A.Perspective(scale=(0.075,0.075),p=1)),
    aug(A.Perspective(scale=(0.10,0.10),p=1))
]),

("Crop & resize",[
    aug(A.RandomResizedCrop(size=(img.shape[0],img.shape[1]),scale=(0.9,0.9),ratio=(1,1),p=1)),
    aug(A.RandomResizedCrop(size=(img.shape[0],img.shape[1]),scale=(0.8,0.8),ratio=(1,1),p=1)),
    aug(A.RandomResizedCrop(size=(img.shape[0],img.shape[1]),scale=(0.7,0.7),ratio=(1,1),p=1))
]),

("Gaussian blur",[
    aug(A.GaussianBlur(blur_limit=(3,3),p=1)),
    aug(A.GaussianBlur(blur_limit=(5,5),p=1)),
    aug(A.GaussianBlur(blur_limit=(7,7),p=1))
]),

("Motion blur",[
    aug(A.MotionBlur(blur_limit=(5,5),p=1)),
    aug(A.MotionBlur(blur_limit=(9,9),p=1)),
    aug(A.MotionBlur(blur_limit=(15,15),p=1))
]),

("Brightness & contrast",[
    aug(A.RandomBrightnessContrast(brightness_limit=(0.1,0.1),contrast_limit=(0.1,0.1),p=1)),
    aug(A.RandomBrightnessContrast(brightness_limit=(0.2,0.2),contrast_limit=(0.2,0.2),p=1)),
    aug(A.RandomBrightnessContrast(brightness_limit=(0.3,0.3),contrast_limit=(0.3,0.3),p=1))
]),

("CLAHE",[
    aug(A.CLAHE(clip_limit=(1.5,1.5),tile_grid_size=(8,8),p=1)),
    aug(A.CLAHE(clip_limit=(3.0,3.0),tile_grid_size=(8,8),p=1)),
    aug(A.CLAHE(clip_limit=(4.0,4.0),tile_grid_size=(8,8),p=1))
]),

("Gaussian noise",[
    aug(A.GaussNoise(var_limit=(10,10),p=1)),
    aug(A.GaussNoise(var_limit=(30,30),p=1)),
    aug(A.GaussNoise(var_limit=(50,50),p=1))
]),

("Local occlusion",[
    aug(A.CoarseDropout(max_holes=1,min_holes=1,max_height=30,min_height=30,max_width=30,min_width=30,fill_value=0,p=1)),
    aug(A.CoarseDropout(max_holes=3,min_holes=3,max_height=50,min_height=50,max_width=50,min_width=50,fill_value=0,p=1)),
    aug(A.CoarseDropout(max_holes=6,min_holes=6,max_height=80,min_height=80,max_width=80,min_width=80,fill_value=0,p=1))
])
]

# ---------- plot ----------
fig,ax=plt.subplots(9,4,figsize=(8.6,16))

for j,t in enumerate(["Original","Low","Medium","High"]):
    ax[0,j].set_title(t,fontsize=13,fontweight="bold",pad=8)

for i,(name,imgs) in enumerate(rows):
    ax[i,0].imshow(img)
    ax[i,0].axis("off")
    ax[i,0].text(-0.08,0.5,name,rotation=90,va="center",ha="center",
                 transform=ax[i,0].transAxes,fontsize=10)
    for j,x in enumerate(imgs,1):
        ax[i,j].imshow(x)
        ax[i,j].axis("off")

plt.subplots_adjust(left=0.12,right=0.995,top=0.97,bottom=0.02,
                    wspace=0.02,hspace=0.08)

plt.savefig(OUT,dpi=600,bbox_inches="tight")
plt.show()
print("Saved:",OUT)