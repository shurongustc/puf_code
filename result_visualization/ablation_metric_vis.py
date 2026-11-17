import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

models = ['PUF-DINO', 'PUF-DINO-Cls', 'DINO-Pretrained', 'EffNet-V2L-Pretrained']
tar = [0.9844, 0.9595, 0.6678, 0.5691]
top1 = [0.9282, 0.8744, 0.3269, 0.0654]
far = [0.0202, 0.0467, 0.3838, 0.4739]
frr = [0.0156, 0.0405, 0.3322, 0.4309]

x = np.arange(4)
w = 0.32
blue, orange = '#4C72B0', '#DD8452'

def label_bars(bars, ax):
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.4f}', (b.get_x()+b.get_width()/2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10,
                    fontname='Arial', fontweight='bold')

def plot(data1, data2, labels, ylabel, filename):
    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    b1 = ax.bar(x-w/2, data1, w, label=labels[0], color=blue)
    b2 = ax.bar(x+w/2, data2, w, label=labels[1], color=orange)

    ax.set_ylabel(ylabel, fontsize=17, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12.5, fontweight='bold',
                       rotation=35, ha='right')
    ax.tick_params(axis='y', labelsize=15)
    for t in ax.get_yticklabels():
        t.set_fontname('Arial')
        t.set_fontweight('bold')

    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper right', frameon=False,
              prop={'family':'Arial', 'size':12, 'weight':'bold'})

    label_bars(b1, ax)
    label_bars(b2, ax)

    plt.subplots_adjust(left=0.13, right=0.98, bottom=0.25, top=0.96)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

plot(tar, top1, ['TAR', 'Top-1 Acc'], 'Score',
     'tar_top1_comparison_4models.png')

plot(far, frr, ['FAR@Threshold', 'FRR@Threshold'], 'Rate',
     'far_frr_comparison_4models.png')