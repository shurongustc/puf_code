# PUF Authentication

This repository contains research code for liquid-crystal physical unclonable function (PUF) image processing and authentication. It includes dataset preparation and augmentation, DINOv2 and EfficientNet fine-tuning, retrieval-based evaluation, visualization utilities, and a Gradio demo.

## Repository layout

- `puf_dataset/` — Dataset splitting, augmentation, and supplementary figure utilities.
- `device_evaluation/` — PUF uniqueness, correlation, histogram, and theoretical capacity analyses.
- `result_visualization/` — Scripts for plotting evaluation metrics and training curves.
- `facebookresearch_dinov2_main/` — Vendored DINOv2 source used by the project scripts.
- Top-level Python files — Model training, evaluation, inference, and demo entry points.

## Requirements

The scripts use Python and commonly depend on:

- PyTorch and TorchVision
- NumPy, Pillow, scikit-learn, tqdm, and Matplotlib
- TensorBoard for training logs
- OpenCV and Albumentations for image augmentation
- pandas, seaborn, and umap-learn for selected analysis and visualization scripts
- Gradio for the web demo

Install a PyTorch/TorchVision build compatible with your operating system and CUDA setup first. Then install the other packages required by the scripts you intend to run. This repository does not currently include a `requirements.txt` or other environment lock file, so package versions are not pinned.

## Dataset format

The training and evaluation scripts expect image filenames to begin with the PUF identity, followed by an underscore. Typical examples are:

```text
<identity>_orig.jpg
<identity>_aug_01.jpg
<identity>_aug_02.jpg
```

The scripts use the filename prefix before the first underscore as the identity label. Triplet-training datasets require at least one original image and one augmented image per identity.

The split utility copies all images belonging to an identity into the same split:

```text
dataset/
├── train/
├── val/
└── test/
```

Edit the example paths at the bottom of [`puf_dataset/puf_data_split.py`](./puf_dataset/puf_data_split.py) before running it:

```bash
python puf_dataset/puf_data_split.py
```

The split ratios default to 80% training, 10% validation, and 10% testing. The script shuffles identities without setting a random seed, so the split may differ between runs.

## Configuration

Most scripts define dataset, checkpoint, DINOv2 source, and output paths directly in the Python file rather than accepting command-line arguments. Before running a script:

1. Update its path constants or paths in its `__main__` block for your environment.
2. Provide the required dataset and checkpoint files.
3. Ensure the output directories are writable.

DINOv2-based scripts add `facebookresearch_dinov2_main` to the Python import path. Fine-tuning scripts also expect a compatible pretrained DINOv2 checkpoint. Some TorchVision workflows may download pretrained EfficientNet weights if they are not already cached.

## Main workflows

### Generate augmented data

The general augmentation and supplementary figure scripts are in `puf_dataset/`. The stress-test generator [`puf_dataset/puf_aug_category2.py`](./puf_dataset/puf_aug_category2.py) creates occlusion and elastic-distortion variants from original images. Update `ORIG_DIR` and `BASE_OUTPUT_DIR` in the script before running:

```bash
python puf_dataset/puf_aug_category2.py
```

### Fine-tune a model

- [`dino_finetune_puf_50epochs.py`](./dino_finetune_puf_50epochs.py) fine-tunes DINOv2 with triplet loss and distributed GPU training.
- [`dino_finetune_classification.py`](./dino_finetune_classification.py) trains the DINOv2 classification variant.
- [`effnet_finetune_puf.py`](./effnet_finetune_puf.py) fine-tunes EfficientNetV2-L with triplet loss and distributed GPU training.

For example, after configuring paths and the required checkpoint:

```bash
python dino_finetune_puf_50epochs.py
```

The DINOv2 and EfficientNet triplet-training scripts are configured for two CUDA processes and use the NCCL backend. Adjust `WORLD_SIZE` and the distributed setup for your available hardware. Training scripts write model checkpoints and may also write CSV logs and TensorBoard event files.

### Evaluate models

Evaluation scripts are selected by model and experiment:

- [`evaluation.py`](./evaluation.py) — DINOv2 retrieval evaluation, ROC plot, and misidentification report.
- [`evaluation_classification.py`](./evaluation_classification.py) — Classification-model evaluation.
- [`evaluation_effnet.py`](./evaluation_effnet.py) and [`evaluation_pretraineeffnet.py`](./evaluation_pretraineeffnet.py) — EfficientNet evaluation.
- [`evaluation_pretrained.py`](./evaluation_pretrained.py) — Evaluation using a pretrained DINOv2 model.
- [`evaluation_ablation.py`](./evaluation_ablation.py), [`evaluation_ablation_four_models.py`](./evaluation_ablation_four_models.py), and [`evaluation_comparison.py`](./evaluation_comparison.py) — Ablation and model comparisons.

Configure each script's dataset, checkpoint, and output paths before running it. Evaluation outputs typically include metrics, plots, and CSV reports in the configured result directory.

### Run inference or the web demo

[`inference.py`](./inference.py) provides `PUFMatcher`, which extracts a feature from an input image and searches an original-image gallery using cosine similarity. Its checkpoint, gallery, cached features, and threshold are configured near the top of the file.

Run the Gradio interface with:

```bash
python app.py
```

The app loads the matcher at startup and listens on port `7860`. The current script sets `share=True`, which requests a shareable Gradio link; change it to `share=False` if the app should remain local.

### Analyze device uniqueness and capacity

Scripts in `device_evaluation/` analyze intra-/inter-PUF similarity, correlation matrices, image histograms, and theoretical encoding capacity. Update their input and output paths before running. For example:

```bash
python device_evaluation/uniqueness.py
python device_evaluation/code_capacity.py
```

Visualization scripts are available in `result_visualization/` and [`visualization.py`](./visualization.py). They may require existing CSV, NPY, image, or checkpoint files as configured in each script.

## Notes

- The scripts are research utilities with configuration embedded in source files; not every script is a command-line application.
- Some paths in the checked-in scripts are machine-specific and must be changed before use.
- Generated datasets, checkpoints, and evaluation results are not included in this repository.
- The DINOv2 source is included under `facebookresearch_dinov2_main/`; consult that directory's upstream license and notices when redistributing or reusing its code.
