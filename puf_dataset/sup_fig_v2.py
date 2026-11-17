import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

# =========================
# Configuration
# =========================
IMG = r"C:\Users\ad\Desktop\puf\paper\puf\puf\puf_dataset\puf_dataset_gray_original\20260105151343592.jpg"
OUT = r"C:\Users\ad\Desktop\puf\paper\puf\puf\puf_dataset\Supplementary_Fig1_perturbations_v2.png"

# =========================
# Load image and resize to 518×518
# =========================
img = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (518, 518), interpolation=cv2.INTER_LINEAR)

def aug(transform):
    return transform(image=img)["image"]

# =========================
# Perturbation examples
# =========================
rows = [

    ("Rotation", [
        aug(A.SafeRotate(limit=(30, 30), border_mode=cv2.BORDER_REFLECT_101, p=1)),
        aug(A.SafeRotate(limit=(90, 90), border_mode=cv2.BORDER_REFLECT_101, p=1)),
        aug(A.SafeRotate(limit=(180, 180), border_mode=cv2.BORDER_REFLECT_101, p=1))
    ]),

    ("Perspective", [
        aug(A.Perspective(scale=(0.05, 0.05), p=1)),
        aug(A.Perspective(scale=(0.075, 0.075), p=1)),
        aug(A.Perspective(scale=(0.10, 0.10), p=1))
    ]),

    ("Crop & resize", [
        aug(A.RandomResizedCrop(size=(518, 518), scale=(0.90, 0.90), ratio=(1, 1), p=1)),
        aug(A.RandomResizedCrop(size=(518, 518), scale=(0.80, 0.80), ratio=(1, 1), p=1)),
        aug(A.RandomResizedCrop(size=(518, 518), scale=(0.70, 0.70), ratio=(1, 1), p=1))
    ]),

    ("Gaussian blur", [
        aug(A.GaussianBlur(blur_limit=(3, 3), p=1)),
        aug(A.GaussianBlur(blur_limit=(5, 5), p=1)),
        aug(A.GaussianBlur(blur_limit=(7, 7), p=1))
    ]),

    ("Brightness & contrast", [
        aug(A.RandomBrightnessContrast(
            brightness_limit=(0.10, 0.10),
            contrast_limit=(0.10, 0.10), p=1)),
        aug(A.RandomBrightnessContrast(
            brightness_limit=(0.20, 0.20),
            contrast_limit=(0.20, 0.20), p=1)),
        aug(A.RandomBrightnessContrast(
            brightness_limit=(0.30, 0.30),
            contrast_limit=(0.30, 0.30), p=1))
    ]),

    ("Gaussian noise", [
        aug(A.GaussNoise(var_limit=(10, 10), p=1)),
        aug(A.GaussNoise(var_limit=(30, 30), p=1)),
        aug(A.GaussNoise(var_limit=(50, 50), p=1))
    ]),

    ("Local occlusion", [
        aug(A.CoarseDropout(
            max_holes=1, min_holes=1,
            max_height=30, min_height=30,
            max_width=30, min_width=30,
            fill_value=0, p=1)),
        aug(A.CoarseDropout(
            max_holes=3, min_holes=3,
            max_height=50, min_height=50,
            max_width=50, min_width=50,
            fill_value=0, p=1)),
        aug(A.CoarseDropout(
            max_holes=6, min_holes=6,
            max_height=80, min_height=80,
            max_width=80, min_width=80,
            fill_value=0, p=1))
    ])

]

# =========================
# Plot: 4 rows × 7 columns
# =========================
fig, ax = plt.subplots(4, 7, figsize=(14, 8))

row_labels = ["Original", "Low", "Medium", "High"]

for col, (label, aug_imgs) in enumerate(rows):

    # Column title
    ax[0, col].set_title(
        label,
        fontsize=10,
        fontweight="bold",
        pad=10
    )

    # Original + three perturbation levels
    images = [img] + aug_imgs

    for r, im in enumerate(images):
        ax[r, col].imshow(im, interpolation="nearest")
        ax[r, col].axis("off")

# Row labels
for r, label in enumerate(row_labels):
    ax[r, 0].text(
        -0.12, 0.5, label,
        transform=ax[r, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

# =========================
# Layout
# =========================
plt.subplots_adjust(
    left=0.065,
    right=0.995,
    top=0.92,
    bottom=0.025,
    wspace=0.05,
    hspace=0.06
)

# =========================
# Save
# =========================
plt.savefig(
    OUT,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
print(f"Saved to: {OUT}")