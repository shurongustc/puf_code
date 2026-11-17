import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ==================== 1. Read training data from the CSV file ====================
# Replace 'puf_training_log.csv' with the path to your CSV file.
csv_file_path = '../puf_training_log.csv'
df = pd.read_csv(csv_file_path)

# Ensure the column names match those in the CSV ('epoch', 'train_loss', 'val_acc', 'val_threshold').
# The code automatically adapts if your CSV contains more rows (e.g., 50 epochs).

# ==================== 2. Set the global font ====================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# ==================== Figure 1: Training loss curve ====================
fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
color_loss = '#e74c3c'

ax1.plot(df['epoch'], df['train_loss'], color=color_loss, marker='o', linewidth=2.5, markersize=6, label='Train loss')
ax1.set_xlabel('Epoch', fontsize=18, fontname='Arial', fontweight='bold')
ax1.set_ylabel('Train Loss', fontsize=18, fontname='Arial', fontweight='bold', color=color_loss)
ax1.tick_params(axis='y', labelcolor=color_loss, labelsize=14)
ax1.tick_params(axis='x', labelsize=14)

for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_weight('bold')

ax1.grid(True, linestyle=':', alpha=0.6)
fig1.tight_layout()
plt.savefig('train_loss_curve.png', dpi=300)
plt.close(fig1)

# ==================== Figure 2: Validation accuracy and threshold ====================
fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=300)
color_acc = '#8e44ad'    # Purple - validation accuracy.
color_thresh = '#3498db' # Blue - validation threshold.

# Set the left axis to validation accuracy (range can be auto-scaled; fixed at 0.5-1.0 here).
line_acc = ax2.plot(df['epoch'], df['val_acc'], color=color_acc, marker='^', linewidth=2.5, markersize=6, label='Validation accuracy')
ax2.set_xlabel('Epoch', fontsize=18, fontname='Arial', fontweight='bold')
ax2.set_ylabel('Validation Accuracy', fontsize=18, fontname='Arial', fontweight='bold', color=color_acc)
ax2.tick_params(axis='y', labelcolor=color_acc, labelsize=14)
ax2.tick_params(axis='x', labelsize=14)
ax2.set_ylim(0.5, 1.0)

# Set the right axis to the validation threshold (range 0.4-0.7).
ax3 = ax2.twinx()
line_thresh = ax3.plot(df['epoch'], df['val_threshold'], color=color_thresh, marker='s', linewidth=2.5, markersize=6, linestyle='--', label='Val threshold')
ax3.set_ylabel('Val Threshold', fontsize=18, fontname='Arial', fontweight='bold', color=color_thresh)
ax3.tick_params(axis='y', labelcolor=color_thresh, labelsize=14)
ax3.set_ylim(0.4, 0.7)

for label in (ax2.get_xticklabels() + ax2.get_yticklabels() + ax3.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_weight('bold')

ax2.grid(True, linestyle=':', alpha=0.6)

# Dynamically mark the epoch with the best validation accuracy (or specify an epoch).
best_row = df.loc[df['val_acc'].idxmax()]
target_epoch = int(best_row['epoch'])
target_thresh = best_row['val_threshold']

ax3.scatter([target_epoch], [target_thresh], s=250, facecolors='none', edgecolors='#d35400', linewidths=3, zorder=5)

# Legend.
lines = line_acc + line_thresh
legend_elements = lines + [Line2D([0], [0], marker='o', color='w', label=f'Best Epoch ({target_epoch})', markeredgecolor='#d35400', markeredgewidth=3, markersize=10)]
legend = ax2.legend(handles=legend_elements, loc='lower left', fontsize=14, framealpha=0.9)
for text in legend.get_texts():
    text.set_fontname('Arial')
    text.set_weight('bold')

fig2.tight_layout()
plt.savefig('val_metrics_curve.png', dpi=300)
plt.show()