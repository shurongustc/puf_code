import numpy as np
import matplotlib.pyplot as plt

# Set the global font to Arial.
plt.rcParams['font.family'] = 'Arial'

# 1. Prepare metric data (use the original FAR and FRR percentages directly).
categories = [
    'Top-1 accuracy', 
    'TAR', 
    'FAR', 
    'FRR', 
    'ROC-AUC'
]

raw_values = [
    92.82,                    # Top-1 Accuracy (%)
    98.44,                    # TAR (%)
    2.0205,                   # FAR (%)
    1.56,                     # FRR (%)
    0.9973 * 100              # Convert ROC AUC to a percentage.
]

values = raw_values.copy()

# 2. Set the radar chart angles.
num_vars = len(categories)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Close the radar chart.
values += values[:1]
angles += angles[:1]

# 3. Start plotting.
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

# Plot the lines and fill colors.
ax.plot(angles, values, linewidth=3, linestyle='solid', label='Fine-tuned DINOv2', color='#1f77b4')
ax.fill(angles, values, color='#1f77b4', alpha=0.2)

# Set the radar chart's angular origin (top is 0 degrees, clockwise).
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Clear the default tick labels.
ax.set_xticks(angles[:-1])
ax.set_xticklabels([]) 

# Set the grid and tick range (0 to 130) to leave enough room for labels.
ax.set_rlim(0, 130)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=16, fontweight='bold', color="grey")

# 4. Precisely position labels at the five vertices (anchored above and below a single baseline).
for angle, raw_val, cat in zip(angles[:-1], raw_values, categories):

    # Format the value labels.
    val_str = f"{raw_val / 100:.4f}" if cat == 'ROC AUC' else f"{raw_val:.2f}%"
    norm_angle = angle % (2 * np.pi)

    # 1. Top (Top-1 Accuracy).
    if np.isclose(norm_angle, 0):
        ax.text(angle, 122, val_str, ha='center', va='bottom', fontsize=18, fontweight='bold', color='#d62728')
        ax.text(angle, 106, cat, ha='center', va='bottom', fontsize=20, fontweight='bold', color='black')

    # 2. Upper right (TAR): red text above black text, left-aligned.
    elif 0 < norm_angle < np.pi / 2:
        r_base = 115
        ax.text(angle, r_base, val_str, ha='left', va='bottom', fontsize=18, fontweight='bold', color='#d62728')
        ax.text(angle, r_base, cat, ha='left', va='top', fontsize=20, fontweight='bold', color='black')

    # 3. Lower right (FAR): black text above red text, left-aligned.
    elif np.pi / 2 <= norm_angle < np.pi:
        r_base = 115
        ax.text(angle, r_base, cat, ha='left', va='bottom', fontsize=20, fontweight='bold', color='black')
        ax.text(angle, r_base, val_str, ha='left', va='top', fontsize=18, fontweight='bold', color='#d62728')

    # 4. Lower left (FRR): black text above red text, right-aligned.
    elif np.pi <= norm_angle < 3 * np.pi / 2:
        r_base = 115
        ax.text(angle, r_base, cat, ha='right', va='bottom', fontsize=20, fontweight='bold', color='black')
        ax.text(angle, r_base, val_str, ha='right', va='top', fontsize=18, fontweight='bold', color='#d62728')

    # 5. Upper left (ROC AUC): red text above black text, right-aligned.
    else:
        r_base = 115
        ax.text(angle, r_base, val_str, ha='right', va='bottom', fontsize=18, fontweight='bold', color='#d62728')
        ax.text(angle, r_base, cat, ha='right', va='top', fontsize=20, fontweight='bold', color='black')

# Save the figure.
plt.tight_layout()
plt.savefig('puf_radar_chart_dinov2_finetuned.png', dpi=300, bbox_inches='tight')
plt.show()