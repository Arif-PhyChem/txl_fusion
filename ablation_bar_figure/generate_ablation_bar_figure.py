import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path("/path/to/3_classes_classification")
OUT = ROOT / "paper_review" / "ablation_comparison_bars.pdf"

models = ["Heuristic", "XGB", "Heur.+XGB", "Heur. + LLM", "TXL"]
overall_colors = ["#7A7A7A", "#2E86AB", "#A23B72"]  # Accuracy, Macro-F1, Weighted-F1
class_metric_colors = ["#4C78A8", "#F58518", "#54A24B"]  # Precision, Recall, F1

overall = {
    "Validation": {
        "Accuracy": [0.686, 0.828, 0.837, 0.8452300405953992, 0.8685723951285521],
        "Macro-F1": [0.597, 0.787, 0.797, 0.8066960168089219, 0.8270282556389672],
        "Weighted-F1": [0.679, 0.836, 0.844, 0.851831231689706, 0.8702907335806618],
    },
    "Held-out test": {
        "Accuracy": [0.678, 0.820, 0.825, 0.8414221981884548, 0.8584561308638637],
        "Macro-F1": [0.595, 0.783, 0.788, 0.8075718089119066, 0.8223959034361293],
        "Weighted-F1": [0.669, 0.828, 0.832, 0.8475233446894259, 0.8601966682594985],
    },
}

classwise = {
    "Validation": {
        "Trivial": {"Precision": [0.771, 0.926, 0.926, 0.9380699088145896, 0.9265922993882691], "Recall": [0.880, 0.855, 0.864, 0.8699788583509513, 0.9073291050035236], "F1": [0.822, 0.889, 0.894, 0.9027422303473491, 0.9168595335588392]},
        "TSM": {"Precision": [0.701, 0.887, 0.892, 0.8919440424505547, 0.8937644341801386], "Recall": [0.577, 0.830, 0.835, 0.8427529626253418, 0.881950774840474], "F1": [0.633, 0.858, 0.863, 0.8666510428872745, 0.8878183069511356]},
        "TI": {"Precision": [0.338, 0.524, 0.547, 0.5625517812758907, 0.6456611570247934], "Recall": [0.333, 0.738, 0.751, 0.7715909090909091, 0.7102272727272727], "F1": [0.336, 0.613, 0.633, 0.6506947771921419, 0.6764069264069265]},
    },
    "Held-out test": {
        "Trivial": {"Precision": [0.763, 0.918, 0.922, 0.9342105263157895, 0.922739244951712], "Recall": [0.877, 0.852, 0.853, 0.8660992907801418, 0.894468085106383], "F1": [0.816, 0.884, 0.886, 0.8988664802002061, 0.9083837510803803]},
        "TSM": {"Precision": [0.679, 0.876, 0.879, 0.887007874015748, 0.8760268857356236], "Recall": [0.567, 0.818, 0.823, 0.8387937453462397, 0.8734177215189873], "F1": [0.618, 0.846, 0.850, 0.8622273249138921, 0.8747203579418344]},
        "TI": {"Precision": [0.364, 0.538, 0.545, 0.5777218376337319, 0.6536098310291859], "Recall": [0.338, 0.734, 0.744, 0.7740303541315345, 0.7175379426644182], "F1": [0.351, 0.621, 0.629, 0.6616216216216216, 0.6840836012861736]},
    },
}

def grouped_bar(ax, data_dict, title, colors, ylabel=False, ylim=(0.25, 1.0), annotate_all=False):
    metric_names = list(data_dict.keys())
    values = np.array([data_dict[k] for k in metric_names])
    x = np.arange(len(models))
    width = 0.22
    for i, metric in enumerate(metric_names):
        offset = (i - (len(metric_names)-1)/2) * width
        bars = ax.bar(x + offset, values[i], width=width, color=colors[i], label=metric, edgecolor='black', linewidth=0.4)
        if annotate_all:
            best_j = int(np.argmax(values[i]))
            for j, bar in enumerate(bars):
                val = values[i][j]
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.008, f"{val:.3f}", ha='center', va='bottom', fontsize=6.8, rotation=90)
                if j == best_j:
                    ax.text(bar.get_x() + bar.get_width()/2, max(val - 0.045, ylim[0] + 0.02), '*', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=8)
    ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    if ylabel:
        ax.set_ylabel('Score', fontsize=9, fontweight='bold')
    ax.tick_params(axis='y', labelsize=8)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

fig, axes = plt.subplots(4, 2, figsize=(13.5, 16))
plt.subplots_adjust(hspace=0.46, wspace=0.18, top=0.92, bottom=0.06)

letters = ['A','B','C','D','E','F','G','H']

# Overall panels
for idx, split in enumerate(["Validation", "Held-out test"]):
    ax = axes[0, idx]
    grouped_bar(ax, overall[split], f"{split}: Overall metrics", overall_colors, ylabel=(idx == 0), ylim=(0.5, 0.9), annotate_all=True)
    ax.text(-0.14, 1.08, letters[idx], transform=ax.transAxes, fontsize=14, fontweight='bold')

# Class-wise panels
panel_map = [
    (1, 0, 'Validation', 'Trivial'),
    (1, 1, 'Held-out test', 'Trivial'),
    (2, 0, 'Validation', 'TSM'),
    (2, 1, 'Held-out test', 'TSM'),
    (3, 0, 'Validation', 'TI'),
    (3, 1, 'Held-out test', 'TI'),
]
for letter, (r, c, split, cls) in zip(letters[2:], panel_map):
    ax = axes[r, c]
    grouped_bar(ax, classwise[split][cls], f"{split}: {cls}\nPrecision / Recall / F1", class_metric_colors, ylabel=(c == 0), ylim=(0.25, 1.0), annotate_all=True)
    ax.text(-0.14, 1.08, letter, transform=ax.transAxes, fontsize=14, fontweight='bold')

from matplotlib.patches import Patch
overall_legend_handles = [
    Patch(facecolor=overall_colors[0], edgecolor='black', linewidth=0.4, label='Accuracy'),
    Patch(facecolor=overall_colors[1], edgecolor='black', linewidth=0.4, label='Macro-F1'),
    Patch(facecolor=overall_colors[2], edgecolor='black', linewidth=0.4, label='Weighted-F1'),
]
class_legend_handles = [
    Patch(facecolor=class_metric_colors[0], edgecolor='black', linewidth=0.4, label='Precision'),
    Patch(facecolor=class_metric_colors[1], edgecolor='black', linewidth=0.4, label='Recall'),
    Patch(facecolor=class_metric_colors[2], edgecolor='black', linewidth=0.4, label='F1-score'),
]
fig.legend(overall_legend_handles, ['Accuracy', 'Macro-F1', 'Weighted-F1'], loc='upper center', ncol=3, frameon=True, fontsize=10, bbox_to_anchor=(0.5, 0.988), borderpad=0.4, handlelength=1.6, columnspacing=1.8)
fig.legend(class_legend_handles, ['Precision', 'Recall', 'F1-score'], loc='upper center', ncol=3, frameon=True, fontsize=10, bbox_to_anchor=(0.5, 0.968), borderpad=0.4, handlelength=1.6, columnspacing=1.8)
fig.savefig(OUT, bbox_inches='tight')
print(f"Wrote {OUT}")
