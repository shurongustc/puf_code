import matplotlib.pyplot as plt
import numpy as np

# Set global font family and size configurations
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Model names and metrics based on user data
models = ['PUF-DINO', 'PUF-EffNet-V2L', 'SIFT']
tar_data = [0.9844, 0.9648, 0.6718]
top1_acc_data = [0.9282, 0.6369, 0.6734]
far_data = [0.0202, 0.0493, 0.0125]
frr_data = [0.0156, 0.0352, 0.3282]

x = np.arange(len(models))
width = 0.35

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontname='Arial', fontweight='bold')

# ----------------- Figure 1: TAR and Top-1 Acc -----------------
fig, ax = plt.subplots(figsize=(8, 6))

rects1 = ax.bar(x - width/2, tar_data, width, label='TAR')
rects2 = ax.bar(x + width/2, top1_acc_data, width, label='Top-1 Acc')

ax.set_ylabel('Score', fontsize=18, fontname='Arial', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=18, fontname='Arial', fontweight='bold', rotation=30, ha='right')
ax.tick_params(axis='y', labelsize=18)
for label in ax.get_yticklabels():
    label.set_fontname('Arial')
    label.set_weight('bold')

ax.legend(fontsize=14, prop={'family': 'Arial', 'size': 14, 'weight': 'bold'})
ax.set_ylim(0, 1.1)
ax.grid(axis='y', linestyle='--', alpha=0.7)

autolabel(rects1, ax)
autolabel(rects2, ax)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('tar_top1_comparison.png', dpi=300)
plt.show()

# ----------------- Figure 2: FAR and FRR -----------------
fig, ax = plt.subplots(figsize=(8, 6))

rects1 = ax.bar(x - width/2, far_data, width, label='FAR@Threshold')
rects2 = ax.bar(x + width/2, frr_data, width, label='FRR@Threshold')

ax.set_ylabel('Rate', fontsize=18, fontname='Arial', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=18, fontname='Arial', fontweight='bold', rotation=30, ha='right')
ax.tick_params(axis='y', labelsize=18)
for label in ax.get_yticklabels():
    label.set_fontname('Arial')
    label.set_weight('bold')

ax.legend(fontsize=14, prop={'family': 'Arial', 'size': 14, 'weight': 'bold'})
ax.set_ylim(0, 1.1)
ax.grid(axis='y', linestyle='--', alpha=0.7)

autolabel(rects1, ax)
autolabel(rects2, ax)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('far_frr_comparison.png', dpi=300)
plt.show()