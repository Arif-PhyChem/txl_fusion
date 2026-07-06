import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

BASE = Path(__file__).resolve().parent
PAPER_REVIEW = Path('/path/to/3_classes_classification/paper_review')

FILES = {
    ("Validation", "XGB"): BASE / "xgb_class_imbalance_summary.csv",
    ("Validation", "TXL Fusion"): BASE / "txl_model_pca50" / "txl_model_pca50_class_imbalance_summary.csv",
    ("Held-out test", "XGB"): BASE / "held_out" / "xgb" / "xgb_held_out_summary.csv",
    ("Held-out test", "TXL Fusion"): BASE / "held_out" / "txl_model_pca50" / "txl_model_pca50_held_out_summary.csv",
}

METRIC_MAP = {
    "val_macro_f1": "Macro-F1",
    "val_weighted_f1": "Weighted-F1",
    "val_accuracy": "Accuracy",
    "val_ti_precision": "TI Precision",
    "val_ti_recall": "TI Recall",
    "val_ti_f1": "TI F1",
    "test_macro_f1": "Macro-F1",
    "test_weighted_f1": "Weighted-F1",
    "test_accuracy": "Accuracy",
    "test_ti_precision": "TI Precision",
    "test_ti_recall": "TI Recall",
    "test_ti_f1": "TI F1",
}

# Keep all metrics for SI
METRIC_ORDER = [
    "Macro-F1",
    "Weighted-F1",
    "Accuracy",
    "TI Precision",
    "TI Recall",
    "TI F1",
]

SETTING_MAP = {
    "baseline": "Unweighted",
    "balanced": "Balanced",
    "balanced_ti_1p5": "Balanced + TI 1.5x",
    "balanced_ti_2p0": "Balanced + TI 2.0x",
    "balanced_ti_3p0": "Balanced + TI 3.0x",
}

SETTING_ORDER = [
    "Unweighted",
    "Balanced",
    "Balanced + TI 1.5x",
    "Balanced + TI 2.0x",
    "Balanced + TI 3.0x",
]

SETTING_SHORT_LABELS = [
    "Unweighted",
    "Balanced",
    "TI 1.5x",
    "TI 2.0x",
    "TI 3.0x",
]

# Explicit colors, markers, and line styles
# Global metrics: solid lines
# TI metrics: dashed lines
METRIC_STYLE = {
    "Macro-F1": {
        "marker": "o",
        "linestyle": "-",
        "color": "#1f77b4",
    },
    "Weighted-F1": {
        "marker": "s",
        "linestyle": "-",
        "color": "#ff7f0e",
    },
    "Accuracy": {
        "marker": "^",
        "linestyle": "-",
        "color": "#2ca02c",
    },
    "TI Precision": {
        "marker": "D",
        "linestyle": "--",
        "color": "#d62728",
    },
    "TI Recall": {
        "marker": "v",
        "linestyle": "--",
        "color": "#9467bd",
    },
    "TI F1": {
        "marker": "P",
        "linestyle": "--",
        "color": "#8c564b",
    },
}


def load_long_scores(path: Path, split: str, model: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "config" not in df.columns:
        raise ValueError(f"Missing required column 'config' in {path}")

    df["Setting"] = df["config"].map(SETTING_MAP)

    # Keep only the five expected settings
    df = df[df["Setting"].isin(SETTING_ORDER)].copy()

    if df.empty:
        raise ValueError(f"No recognized settings found in {path}")

    df["Setting"] = pd.Categorical(
        df["Setting"],
        categories=SETTING_ORDER,
        ordered=True,
    )

    metric_cols = [c for c in df.columns if c in METRIC_MAP]

    if not metric_cols:
        raise ValueError(f"No recognized metric columns found in {path}")

    scores = df[["Setting", *metric_cols]].rename(columns=METRIC_MAP)

    available_metrics = [m for m in METRIC_ORDER if m in scores.columns]

    if not available_metrics:
        raise ValueError(f"No requested metrics available in {path}")

    scores = scores[["Setting", *available_metrics]]

    long = scores.melt(
        id_vars="Setting",
        var_name="Metric",
        value_name="Score",
    )

    long["Split"] = split
    long["Model"] = model

    return long


def build_plot_dataframe() -> pd.DataFrame:
    parts = []

    for (split, model), path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        parts.append(load_long_scores(path, split, model))

    data = pd.concat(parts, ignore_index=True)

    data["Metric"] = pd.Categorical(
        data["Metric"],
        categories=METRIC_ORDER,
        ordered=True,
    )

    data["Setting"] = pd.Categorical(
        data["Setting"],
        categories=SETTING_ORDER,
        ordered=True,
    )

    return data


def plot_metric_sweep(ax, data: pd.DataFrame, split: str, model: str):
    panel_data = data[
        (data["Split"] == split) &
        (data["Model"] == model)
    ].copy()

    x_positions = list(range(len(SETTING_ORDER)))

    for metric in METRIC_ORDER:
        sub = panel_data[panel_data["Metric"] == metric].copy()

        if sub.empty:
            continue

        sub = (
            sub.set_index("Setting")
            .reindex(SETTING_ORDER)
            .reset_index()
        )

        style = METRIC_STYLE[metric]

        ax.plot(
            x_positions,
            sub["Score"],
            label=metric,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=2.2,
            markersize=6.0,
        )

    ax.set_title(f"{split}: {model}", fontsize=12, weight="semibold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(SETTING_SHORT_LABELS, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.30, 1.00)

    ax.grid(axis="y", alpha=0.25, linewidth=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_custom_legend_handles():
    legend_handles = []

    for metric in METRIC_ORDER:
        style = METRIC_STYLE[metric]

        legend_handles.append(
            Line2D(
                [0],
                [0],
                label=metric,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                linewidth=2.6,
                markersize=6.8,
            )
        )

    return legend_handles


def main():
    data = build_plot_dataframe()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.5, 9.5),
        sharey=True,
        constrained_layout=True,
    )

    panel_order = [
        ("Validation", "XGB"),
        ("Validation", "TXL Fusion"),
        ("Held-out test", "XGB"),
        ("Held-out test", "TXL Fusion"),
    ]

    for ax, (split, model) in zip(axes.flatten(), panel_order):
        plot_metric_sweep(ax, data, split, model)
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")

    legend_handles = make_custom_legend_handles()

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.045),
        fontsize=9,
        handlelength=3.5,
        handletextpad=0.7,
        columnspacing=1.2,
    )

    fig.suptitle(
        "Sensitivity of validation and held-out test performance to class weighting",
        fontsize=14,
        weight="semibold",
        y=1.09,
    )

    out_pdf = PAPER_REVIEW / "class_imbalance.pdf"
    out_png = PAPER_REVIEW / "class_imbalance.png"

    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    print(out_pdf)
    print(out_png)


if __name__ == "__main__":
    main()