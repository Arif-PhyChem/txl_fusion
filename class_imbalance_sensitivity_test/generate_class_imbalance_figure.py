import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from matplotlib.gridspec import GridSpec

BASE = Path(__file__).resolve().parent
PAPER_REVIEW = Path('/path/to/3_classes_classification/paper_review')

VALIDATION_FILES = {
    "XGB": BASE / "xgb_class_imbalance_summary.csv",
    "TXL Fusion": BASE / "txl" / "txl_class_imbalance_summary.csv",
}
TEST_FILES = {
    "XGB": BASE / "held_out" / "xgb" / "xgb_held_out_summary.csv",
    "TXL Fusion": BASE / "held_out" / "txl" / "txl_held_out_summary.csv",
}

METRICS = [
    ("TI Recall", "ti_recall"),
    ("TI F1", "ti_f1"),
    ("Macro-F1", "macro_f1"),
    ("Accuracy", "accuracy"),
]

MODEL_STYLE = {
    "XGB": {"color": "#1f77b4", "linestyle": "--"},
    "TXL Fusion": {"color": "#d62728", "linestyle": "-"},
}


def load_summary(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.set_index("config")
    cols = {
        f"{prefix}_ti_recall": "ti_recall",
        f"{prefix}_ti_f1": "ti_f1",
        f"{prefix}_macro_f1": "macro_f1",
        f"{prefix}_accuracy": "accuracy",
    }
    return df[list(cols)].rename(columns=cols)


def annotate_point(ax, x, y, text, color, dy):
    ax.text(x, y + dy, text, color=color, fontsize=8, ha="center", va="bottom")


def add_slope_panel(ax, title, summaries):
    xs = list(range(len(METRICS)))
    offsets = {"XGB": -0.10, "TXL Fusion": 0.10}
    for model, df in summaries.items():
        style = MODEL_STYLE[model]
        base = df.loc["baseline"]
        bal = df.loc["balanced"]
        for i, (label, key) in enumerate(METRICS):
            x0 = xs[i] + offsets[model] - 0.05
            x1 = xs[i] + offsets[model] + 0.05
            y0 = float(base[key])
            y1 = float(bal[key])
            ax.plot([x0, x1], [y0, y1], color=style["color"], linestyle=style["linestyle"], linewidth=2.0, zorder=2)
            ax.scatter([x0], [y0], s=50, facecolors='white', edgecolors=style["color"], linewidths=1.8, zorder=3)
            ax.scatter([x1], [y1], s=58, facecolors=style["color"], edgecolors=style["color"], linewidths=1.0, zorder=3)
            annotate_point(ax, x0, y0, f"{y0:.3f}", style["color"], 0.008)
            annotate_point(ax, x1, y1, f"{y1:.3f}", style["color"], 0.008)
    ax.set_xticks(xs)
    ax.set_xticklabels([m[0] for m in METRICS], rotation=0)
    ax.set_ylim(0.44, 0.89)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=12, weight='semibold')
    ax.grid(axis='y', alpha=0.25, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_delta_panel(ax, val_summaries, test_summaries):
    xs = list(range(len(METRICS)))
    width = 0.17
    positions = {
        ("Validation", "XGB"): -1.5 * width,
        ("Validation", "TXL Fusion"): -0.5 * width,
        ("Held-out test", "XGB"): 0.5 * width,
        ("Held-out test", "TXL Fusion"): 1.5 * width,
    }
    dataset_alpha = {"Validation": 0.95, "Held-out test": 0.55}
    for dataset_name, summaries in [("Validation", val_summaries), ("Held-out test", test_summaries)]:
        for model, df in summaries.items():
            style = MODEL_STYLE[model]
            delta = df.loc["balanced"] - df.loc["baseline"]
            ys = [float(delta[key]) for _, key in METRICS]
            bars = ax.bar([x + positions[(dataset_name, model)] for x in xs], ys, width=width, color=style["color"], alpha=dataset_alpha[dataset_name], edgecolor=style["color"], linewidth=1.0)
            for bar, y in zip(bars, ys):
                va = 'bottom' if y >= 0 else 'top'
                offset = 0.004 if y >= 0 else -0.004
                ax.text(bar.get_x() + bar.get_width()/2, y + offset, f"{y:+.3f}", ha='center', va=va, fontsize=8, color=style["color"])
    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([m[0] for m in METRICS])
    ax.set_ylabel(r"$\Delta$ score (Balanced $-$ Baseline)")
    ax.set_title("C. Balanced-weight effect generalizes from validation to held-out test", fontsize=12, weight='semibold')
    ax.grid(axis='y', alpha=0.25, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    val_summaries = {model: load_summary(path, "val") for model, path in VALIDATION_FILES.items()}
    test_summaries = {model: load_summary(path, "test") for model, path in TEST_FILES.items()}

    fig = plt.figure(figsize=(13.5, 10.0), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    add_slope_panel(ax1, "A. Validation set: baseline to balanced", val_summaries)
    add_slope_panel(ax2, "B. Held-out test set: baseline to balanced", test_summaries)
    add_delta_panel(ax3, val_summaries, test_summaries)

    legend_items = [
        Line2D([0], [0], color=MODEL_STYLE["XGB"]["color"], linestyle=MODEL_STYLE["XGB"]["linestyle"], marker='o', markerfacecolor='white', markeredgecolor=MODEL_STYLE["XGB"]["color"], linewidth=2, label='XGB'),
        Line2D([0], [0], color=MODEL_STYLE["TXL Fusion"]["color"], linestyle=MODEL_STYLE["TXL Fusion"]["linestyle"], marker='o', markerfacecolor='white', markeredgecolor=MODEL_STYLE["TXL Fusion"]["color"], linewidth=2, label='TXL Fusion'),
        Line2D([0], [0], color='black', linestyle='None', marker='o', markerfacecolor='white', markeredgecolor='black', markersize=7, label='Baseline'),
        Line2D([0], [0], color='black', linestyle='None', marker='o', markerfacecolor='black', markeredgecolor='black', markersize=7, label='Balanced'),
        Line2D([0], [0], color='gray', linewidth=8, alpha=0.95, label='Validation Δ'),
        Line2D([0], [0], color='gray', linewidth=8, alpha=0.55, label='Held-out test Δ'),
    ]
    fig.legend(handles=legend_items, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))

    out = PAPER_REVIEW / 'class_imbalance_validation_test_consistency.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(out)


if __name__ == '__main__':
    main()
