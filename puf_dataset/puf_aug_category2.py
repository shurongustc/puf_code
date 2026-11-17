import albumentations as A
import cv2
import os
import numpy as np
from tqdm import tqdm

# === 1. Configuration ===
ORIG_DIR = './puf_splitted_dataset/test' 
BASE_OUTPUT_DIR = './puf_splitted_dataset/puf_category_stress_test'

# Redefined configuration.
# Occlusion level is the number of occlusion patches; the total area is fixed at 0.1.
STRESS_CONFIG = {
    "Occlusion": [1, 2, 3],           # Test the number of occlusion patches.
    "Refraction": [10, 20, 30, 40, 50] # Simulated refraction distortion strength (ElasticTransform).
}

def generate_custom_stress_datasets():
    orig_files = [f for f in os.listdir(ORIG_DIR) if f.lower().endswith('_orig.jpg')]
    if not orig_files:
        print("❌ No original images found.")
        return

    for category, values in STRESS_CONFIG.items():
        for val in values:
            save_dir = os.path.join(BASE_OUTPUT_DIR, f"{category}_level_{val}")
            
            # Skip if already generated.
            if os.path.exists(save_dir) and len(os.listdir(save_dir)) >= len(orig_files):
                print(f"✅ Skipping existing: {category} Level {val}")
                continue
                
            os.makedirs(save_dir, exist_ok=True)
            print(f"🚀 Generating: {category} Level {val}...")

            if category == "Occlusion":
                # Here, val represents the number of patches N.
                num_holes = val
                total_area_ratio = 0.1
                # Calculate each patch's side length to keep the total area fixed.
                side = int(np.sqrt(total_area_ratio / num_holes) * 518)
                
                aug = A.Compose([
                    A.CoarseDropout(
                        max_holes=num_holes, 
                        min_holes=num_holes, 
                        max_height=side, 
                        min_height=side, 
                        max_width=side, 
                        min_width=side, 
                        fill_value=0, 
                        p=1.0
                    ),
                    A.Resize(518, 518)
                ])
            
            elif category == "Refraction":
                # Use an elastic transform to simulate local distortions caused by light passing through a transparent medium (e.g., glue or glass).
                aug = A.Compose([
                    A.ElasticTransform(
                        alpha=val,       # Deformation intensity.
                        sigma=val * 0.5, # Deformation smoothness.
                        p=1.0, 
                        border_mode=cv2.BORDER_REFLECT_101
                    ),
                    A.Resize(518, 518)
                ])

            for fname in orig_files:
                img_path = os.path.join(ORIG_DIR, fname)
                image = cv2.imread(img_path)
                if image is None: continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply augmentation.
                augmented = aug(image=image)['image']
                
                name_base = fname.replace('_orig.jpg', '')
                save_path = os.path.join(save_dir, f"{name_base}_aug.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    generate_custom_stress_datasets()
    print("\n✨ Occlusion and refraction experiment datasets generated.")