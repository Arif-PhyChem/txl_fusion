import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
# === Load Topogivities ===
SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_REVIEW_DIR = SCRIPT_DIR.parents[1] / 'paper_review'

ARTIFACT_DIR = SCRIPT_DIR / 'weighted_topogivity_artifacts'

with (ARTIFACT_DIR / 'trivial_topogivities.json').open('r') as f:
    trivial_topogivities = json.load(f)

with (ARTIFACT_DIR / 'sm_topogivities.json').open('r') as f:
    sm_topogivities = json.load(f)

with (ARTIFACT_DIR / 'ti_topogivities.json').open('r') as f:
    ti_topogivities = json.load(f)

# === Periodic Table Layout ===
periodic_layout = {
    'H': (0, 0), 'He': (0, 17),
    'Li': (1, 0), 'Be': (1, 1), 'B': (1, 12), 'C': (1, 13), 'N': (1, 14), 'O': (1, 15), 'F': (1, 16), 'Ne': (1, 17),
    'Na': (2, 0), 'Mg': (2, 1), 'Al': (2, 12), 'Si': (2, 13), 'P': (2, 14), 'S': (2, 15), 'Cl': (2, 16), 'Ar': (2, 17),
    'K': (3, 0), 'Ca': (3, 1), 'Sc': (3, 2), 'Ti': (3, 3), 'V': (3, 4), 'Cr': (3, 5), 'Mn': (3, 6), 'Fe': (3, 7),
    'Co': (3, 8), 'Ni': (3, 9), 'Cu': (3, 10), 'Zn': (3, 11), 'Ga': (3, 12), 'Ge': (3, 13), 'As': (3, 14), 'Se': (3, 15),
    'Br': (3, 16), 'Kr': (3, 17),
    'Rb': (4, 0), 'Sr': (4, 1), 'Y': (4, 2), 'Zr': (4, 3), 'Nb': (4, 4), 'Mo': (4, 5), 'Tc': (4, 6), 'Ru': (4, 7),
    'Rh': (4, 8), 'Pd': (4, 9), 'Ag': (4, 10), 'Cd': (4, 11), 'In': (4, 12), 'Sn': (4, 13), 'Sb': (4, 14), 'Te': (4, 15),
    'I': (4, 16), 'Xe': (4, 17),
    'Cs': (5, 0), 'Ba': (5, 1), 'La': (7, 3), 'Ce': (7, 4), 'Pr': (7, 5), 'Nd': (7, 6), 'Pm': (7, 7), 'Sm': (7, 8),
    'Eu': (7, 9), 'Gd': (7, 10), 'Tb': (7, 11), 'Dy': (7, 12), 'Ho': (7, 13), 'Er': (7, 14), 'Tm': (7, 15), 'Yb': (7, 16), 'Lu': (7, 17),
    '57-71': (5, 2), 'Hf': (5, 3), 'Ta': (5, 4), 'W': (5, 5), 'Re': (5, 6), 'Os': (5, 7), 'Ir': (5, 8), 'Pt': (5, 9),
    'Au': (5, 10), 'Hg': (5, 11), 'Tl': (5, 12), 'Pb': (5, 13), 'Bi': (5, 14), 'Po': (5, 15), 'At': (5, 16), 'Rn': (5, 17),
    'Fr': (6, 0), 'Ra': (6, 1), 'Ac': (8, 3), 'Th': (8, 4), 'Pa': (8, 5), 'U': (8, 6), 'Np': (8, 7), 'Pu': (8, 8),
    'Am': (8, 9), 'Cm': (8, 10), 'Bk': (8, 11), 'Cf': (8, 12), 'Es': (8, 13), 'Fm': (8, 14), 'Md': (8, 15), 'No': (8, 16), 'Lr': (8, 17),
    '89-103': (6, 2), 'Rf': (6, 3), 'Db': (6, 4), 'Sg': (6, 5), 'Bh': (6, 6), 'Hs': (6, 7), 'Mt': (6, 8),
    'Ds': (6, 9), 'Rg': (6, 10), 'Cn': (6, 11), 'Nh': (6, 12), 'Fl': (6, 13), 'Mc': (6, 14), 'Lv': (6, 15), 'Ts': (6, 16),
    'Og': (6, 17)
}


cmap = cm.Oranges #YlOrBr  # or another diverging colormap
CELL_H = 0.52

def y0(row):
    return -row * CELL_H

def plot_topogivity_subplot(ax, data, title):
    # === Normalize topogivity range ===
    min_val = min(data.values())
    max_val = max(data.values())
    
    # Colormap and normalization
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    
    # === Plotting ===
    ax.set_xlim(-1, 19)
    ax.set_ylim(-8.8 * CELL_H, 0.9)
    ax.set_aspect('auto')
    ax.axis('off')
    
    
    for el, (row, col) in periodic_layout.items():
        if el == '57-71':
            ax.add_patch(plt.Rectangle((col, y0(row)), 1, CELL_H, edgecolor='black', facecolor='lightblue', linewidth=0.45))
            #ax.text(col + 0.5, -row + 0.65, '57–71', ha='center', va='center', fontsize=10, weight='bold', color='black')
            ax.text(col + 0.5, y0(row) + 0.38, 'Lan', ha='center', va='center', fontsize=6.7, color='black', weight='bold')
    
        elif el == '89-103':
            ax.add_patch(plt.Rectangle((col, y0(row)), 1, CELL_H, edgecolor='black', facecolor='lightblue', linewidth=0.45))
            #ax.text(col + 0.5, -row + 0.65, '89–103', ha='center', va='center', fontsize=10, weight='bold', color='black')
            ax.text(col + 0.5, y0(row) + 0.38, 'Act', ha='center', va='center', fontsize=6.7, color='black', weight='bold')
    
        elif el not in ['57-71', '89-103']:  # this avoids falling into else
            val = data.get(el, None)
            facecolor = cmap(norm(val)) if val is not None else 'white'
    
            ax.add_patch(plt.Rectangle((col, y0(row)), 1, CELL_H, edgecolor='black', facecolor=facecolor, linewidth=0.45))
            ax.text(col + 0.5, y0(row) + 0.38, el, ha='center', va='center', fontsize=6.9, weight='bold', color='black')
    
            if val is not None:
                ax.text(col + 0.5, y0(row) + 0.15, f"{val:.2f}", ha='center', va='center', fontsize=5.9, color='black')

    # Colorbar
    # === Create a short, centered colorbar on the right ===
    pos = ax.get_position()  # Get subplot position
    cbar_height = 0.125      # Compact colorbar height for stacked layout
    cbar_bottom = pos.y0 + (pos.height - cbar_height) / 2  # Center vertically

    cax = fig.add_axes([
        pos.x1 + 0.01,    # x: just to the right of subplot
        cbar_bottom,      # y: centered vertically
        0.015,            # width of colorbar
        cbar_height       # height of colorbar
    ])
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    #cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.005, fraction=0.02, shrink=0.5)#, aspect=30)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    #cbar.set_label(r"$\tau$ Value", fontsize=14)
    #cbar.ax.set_position([0.05, 0.4, 0.7, 0.01])  # [left, bottom, width, height]
    
    ax.text(5.9, y0(1) + 0.36, title, ha='center', va='center', fontsize=9.8, weight='bold')




def print_metric_summary():
    metadata_path = ARTIFACT_DIR / 'metadata.json'
    validation_path = ARTIFACT_DIR / 'validation_metrics.json'
    test_path = ARTIFACT_DIR / 'heldout_test_metrics.json'
    if not (metadata_path.exists() and validation_path.exists() and test_path.exists()):
        return

    metadata = json.loads(metadata_path.read_text())
    validation = json.loads(validation_path.read_text())
    test = json.loads(test_path.read_text())

    def fmt(metrics):
        ti = metrics['classification_report']['topological']
        return (
            f"Acc={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}, "
            f"Weighted-F1={metrics['weighted_f1']:.4f}, "
            f"TI P/R/F1={ti['precision']:.4f}/{ti['recall']:.4f}/{ti['f1-score']:.4f}"
        )

    print(
        f"Selected balanced topogivity gamma={metadata['selected_gamma']:.2e}, "
        f"C={metadata['selected_C']:.4g}"
    )
    print(f"Validation three-class argmax: {fmt(validation)}")
    print(f"Held-out three-class argmax: {fmt(test)}")

# Enable LaTeX rendering for text in Matplotlib
#plt.rcParams['text.usetex'] = True

# === Use explicit axes positions for compact, non-overlapping panels ===
fig = plt.figure(figsize=(7.3, 8.6), constrained_layout=False)

ax1 = fig.add_axes([0.02, 0.685, 0.84, 0.285])
ax2 = fig.add_axes([0.02, 0.365, 0.84, 0.285])
ax3 = fig.add_axes([0.02, 0.045, 0.84, 0.285])

# Plot each subplot
plot_topogivity_subplot(ax1, trivial_topogivities, r"Element-wise $\tau_E$ for Trivials")
plot_topogivity_subplot(ax2, sm_topogivities, r"Element-wise $\tau_E$ for TSMs")
plot_topogivity_subplot(ax3, ti_topogivities, r"Element-wise $\tau_E$ for TIs")

# Adjust layout
#plt.tight_layout()
#plt.subplots_adjust(hspace=-0.9)  # Extra space between subplots

# Save and show
out_path = ARTIFACT_DIR / 'tau_periodic_table.pdf'
plt.savefig(out_path, format='pdf', bbox_inches='tight')
PAPER_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(PAPER_REVIEW_DIR / 'tau_periodic_table.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)
print(f'Wrote {out_path}')
print(f"Wrote {PAPER_REVIEW_DIR / 'tau_periodic_table.pdf'}")
print_metric_summary()
