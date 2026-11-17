import os
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola

# ================= Configuration =================
TEST_DIR = r"C:\Users\ad\Desktop\puf\puf_demo_code\puf_dataset\puf_splitted_dataset\test"
OUTPUT_DIR = './segmentation_comparison_batch'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIZE = 518
POLARITY = 'dark'       # dark or bright
SAUVOLA_WINDOW = 51
SAUVOLA_K = 0.2
NUM_SAMPLES = 10        # Total number of images to evaluate for global statistics
RANDOM_SEED = 42        # Fixed random seed for reproducibility

# Specify the target image filename that MUST be included in the evaluation
TYPICAL_SAMPLE_FILENAME = "20251230145930368_orig.jpg"
# =================================================

def set_seed(seed):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def preprocess(img):
    h, w = img.shape
    scale = SIZE / max(h, w)
    nw, nh = round(w * scale), round(h * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    return cv2.copyMakeBorder(
        img, top, SIZE - nh - top, left, SIZE - nw - left,
        cv2.BORDER_REFLECT_101
    )

def process_single_image(img_path):
    """Process a single image with 4 binarization methods and extract Gabor feature grays."""
    img = preprocess(np.array(Image.open(img_path).convert('L')))
    inverse = POLARITY == 'dark'

    # 1. Fixed threshold
    fixed = img < 127 if inverse else img >= 127

    # 2. Otsu
    mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, otsu = cv2.threshold(img, 0, 255, mode + cv2.THRESH_OTSU)
    otsu = otsu > 0

    # 3. Sauvola
    t = threshold_sauvola(img, window_size=SAUVOLA_WINDOW, k=SAUVOLA_K)
    sauvola = img < t if inverse else img > t

    # 4. Gabor filter bank + Otsu
    gray = img.astype(np.float32) / 255
    gray -= cv2.GaussianBlur(gray, (0, 0), 5)

    responses = []
    for theta in np.arange(8) * np.pi / 8:
        kernel = cv2.getGaborKernel((31, 31), 3, theta, 8, 0.5, 0, ktype=cv2.CV_32F)
        kernel -= kernel.mean()
        kernel /= np.sum(np.abs(kernel)) + 1e-8
        responses.append(np.abs(cv2.filter2D(gray, -1, kernel)))

    gabor_response = np.max(responses, axis=0)
    lo, hi = np.percentile(gabor_response, [1, 99])
    gabor_response = np.clip((gabor_response - lo) / (hi - lo + 1e-8), 0, 1)
    gabor_uint8 = (gabor_response * 255).astype(np.uint8)

    _, gabor = cv2.threshold(gabor_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gabor = gabor > 0

    # Densities for all methods
    densities = {
        'Fixed 127': fixed.mean(),
        'Otsu': otsu.mean(),
        'Sauvola': sauvola.mean(),
        'Gabor + Otsu': gabor.mean()
    }
    
    # Images for 5-in-1 comparison plot
    results_images = {
        'Original': img,
        'Fixed 127': fixed,
        'Otsu': otsu,
        'Sauvola': sauvola,
        'Gabor + Otsu': gabor
    }

    # Extract original grayscale values exclusively from Gabor + Otsu feature mask
    if gabor.sum() > 0:
        gabor_feature_grays = img[gabor].astype(np.int32)
    else:
        gabor_feature_grays = np.array([], dtype=np.int32)

    return densities, gabor_feature_grays, results_images

def main():
    # Fix random seed
    set_seed(RANDOM_SEED)

    # Filter files containing '_orig' and ending with images
    all_files = sorted([f for f in os.listdir(TEST_DIR) if '_orig' in f and f.endswith(('.jpg', '.png'))])
    
    if len(all_files) == 0:
        print(f"Error: No '_orig' images found in {TEST_DIR}!")
        return

    # Ensure the specified typical sample is included in the evaluation list
    selected_files = []
    if TYPICAL_SAMPLE_FILENAME in all_files:
        selected_files.append(TYPICAL_SAMPLE_FILENAME)
    else:
        print(f"Warning: Specified file '{TYPICAL_SAMPLE_FILENAME}' not found in directory. Proceeding with random selection.")

    # Fill the remaining quota randomly
    remaining_files = [f for f in all_files if f not in selected_files]
    needed = min(NUM_SAMPLES - len(selected_files), len(remaining_files))
    if needed > 0:
        selected_files.extend(random.sample(remaining_files, needed))

    print(f"🎯 [Random Seed: {RANDOM_SEED}] Selected {len(selected_files)} '_orig' images for evaluation (including typical sample):\n")
    for sf in selected_files:
        print(f"  - {sf}")

    density_sums = {'Fixed 127': 0.0, 'Otsu': 0.0, 'Sauvola': 0.0, 'Gabor + Otsu': 0.0}
    all_gabor_feature_grays = []
    
    sample_results = None
    sample_name = ""

    for i, f in enumerate(tqdm(selected_files, desc='Processing Images')):
        img_path = os.path.join(TEST_DIR, f)
        densities, gabor_grays, res_imgs = process_single_image(img_path)
        
        for method in density_sums:
            density_sums[method] += densities[method]
        
        if len(gabor_grays) > 0:
            all_gabor_feature_grays.append(gabor_grays)

        # Capture the specified typical sample for the comparison plot
        if f == TYPICAL_SAMPLE_FILENAME:
            sample_results = res_imgs
            sample_name = os.path.splitext(f)[0]

    # Fallback if typical sample wasn't set for some reason
    if sample_results is None:
        first_path = os.path.join(TEST_DIR, selected_files[0])
        _, _, sample_results = process_single_image(first_path)
        sample_name = os.path.splitext(selected_files[0])[0]

    # 1. Print Average Pattern Filling Densities for all methods
    print("\n" + "="*55)
    print(f"=== Average Pattern Filling Density ({len(selected_files)} Images) ===")
    print("="*55)
    for method in density_sums:
        avg_d = density_sums[method] / len(selected_files)
        print(f"{method:15s}: Average D = {avg_d:.4f}")
    print("="*55)

    # 2. Compute Valid Grayscale Range strictly using Gabor + Otsu features (1% ~ 99% CDF)
    combined_grays = np.concatenate(all_gabor_feature_grays) if len(all_gabor_feature_grays) > 0 else np.array([])
    
    if len(combined_grays) > 0:
        hist, _ = np.histogram(combined_grays, bins=256, range=(0, 256))
        total_count = np.sum(hist)
        cdf = np.cumsum(hist) / total_count
        
        valid_low_idxs = np.where(cdf >= 0.01)[0]
        valid_high_idxs = np.where(cdf <= 0.99)[0]
        
        valid_low = int(valid_low_idxs[0]) if len(valid_low_idxs) > 0 else 0
        valid_high = int(valid_high_idxs[-1]) if len(valid_high_idxs) > 0 else 255
        valid_c = valid_high - valid_low + 1
    else:
        valid_low, valid_high, valid_c = 0, 0, 0
        hist = np.zeros(256)

    print("\n" + "="*55)
    print(f"=== Valid Grayscale Range (1% ~ 99% CDF) for Gabor + Otsu ===")
    print("="*55)
    print(f"  * Valid Low (C_min): {valid_low}")
    print(f"  * Valid High (C_max): {valid_high}")
    print(f"  * Effective Grayscale Levels (C): {valid_c}")
    print("="*55)

    # Plot 1: Gabor + Otsu Feature Grayscale Distribution & Valid Range
    plt.figure(figsize=(9, 6), dpi=300)
    pdf = hist / np.sum(hist) if np.sum(hist) > 0 else hist
    plt.plot(pdf, color='#2b5c8f', linewidth=2.5, label='Gabor+Otsu Feature PDF')
    plt.axvline(valid_low, color='#d9534f', linestyle='--', linewidth=2, label=f'Valid Low: {valid_low}')
    plt.axvline(valid_high, color='#d9534f', linestyle='--', linewidth=2, label=f'Valid High: {valid_high}')
    plt.axvspan(valid_low, valid_high, color='#f0ad4e', alpha=0.2, label=f'Effective Range (C = {valid_c})')
    
    plt.title(f'Valid Grayscale Distribution (Gabor + Otsu Features)', fontsize=16, fontweight='bold')
    plt.xlabel('Grayscale Intensity (0 - 255)', fontsize=14)
    plt.ylabel('Normalized Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    dist_plot_path = os.path.join(OUTPUT_DIR, 'gabor_otsu_valid_grayscale_dist.png')
    plt.savefig(dist_plot_path)
    plt.close()
    print(f"📈 Grayscale distribution plot saved to: {dist_plot_path}")

    # Plot 2: High-Definition 5-in-1 Comparison Figure with Larger Fonts (>= 20)
    if sample_results is not None:
        fig, axes = plt.subplots(1, 5, figsize=(25, 5), dpi=300)
        
        results_list = [
            ('Original', sample_results['Original'], None),
            ('Fixed 127', sample_results['Fixed 127'], sample_results['Fixed 127'].mean()),
            ('Otsu', sample_results['Otsu'], sample_results['Otsu'].mean()),
            ('Sauvola', sample_results['Sauvola'], sample_results['Sauvola'].mean()),
            ('Gabor + Otsu', sample_results['Gabor + Otsu'], sample_results['Gabor + Otsu'].mean())
        ]

        for ax, (name, result, density) in zip(axes, results_list):
            ax.imshow(result, cmap='gray', vmin=0, vmax=255 if name == 'Original' else 1)
            title = name if density is None else f'{name}\nD={density:.4f}'
            # Font size set to 20 for crystal clear and large titles
            ax.set_title(title, fontsize=20, fontweight='bold', pad=10)
            ax.axis('off')

        plt.tight_layout()
        comparison_plot_path = os.path.join(OUTPUT_DIR, f"{sample_name}_comparison.png")
        plt.savefig(comparison_plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"🖼️ High-definition 5-in-1 comparison plot for '{sample_name}' saved to: {comparison_plot_path}")

if __name__ == '__main__':
    main()