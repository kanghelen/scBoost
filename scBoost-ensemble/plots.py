"""
Plot Top-1 and Top-5 accuracy strip plots by Combination Type.

Usage:
    python plot_strip.py --input summary_metrics_slim.tsv
    python plot_strip.py --input summary_metrics_slim.tsv --out fig.svg --dpi 600
"""
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd

FONT_RC = {
    "svg.fonttype":          "none",
    "pdf.fonttype":          42,
    "font.family":           "sans-serif",
    "font.size":             6,
    "axes.titlesize":        7,
    "axes.labelsize":        6,
    "xtick.labelsize":       6,
    "ytick.labelsize":       6,
    "legend.fontsize":       6,
    "legend.title_fontsize": 6,
    "savefig.bbox":          "tight",
}

PALETTE      = {"classic": "gray", "scFM": "#38b4f7", "scFM_full": "#0141a8"}
COMBO_ORDER  = ["single", "concat", "stacked", "stacked_perm"]
COMBO_LABELS = ["Single", "Concat", "Stacked", "Stacked\nPerm"]
METRICS      = [("top1_accuracy", "Top-1 Accuracy"), ("top5_accuracy", "Top-5 Accuracy")]

def main(input_path, out_path, dpi=300):
    df = pd.read_csv(input_path, sep="\t").rename(columns={
        "Combination/Model Type": "combo", "Embedding Type": "embedding",
        "Methods": "methods", "Split": "split"})
    df["combo"] = pd.Categorical(df["combo"], categories=COMBO_ORDER, ordered=True)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(FONT_RC)

    fig, axes = plt.subplots(2, 2, figsize=(3, 3), sharey="row", sharex=True)

    for row, (metric, ylabel) in enumerate(METRICS):
        for col, split in enumerate(["stratified", "holdout"]):
            ax = axes[row, col]
            sub = df[df["split"] == split]
            sns.stripplot(data=sub, x="combo", y=metric, hue="embedding",
                          palette=PALETTE, order=COMBO_ORDER, dodge=True,
                          jitter=0.15, size=1.5, alpha=0.75, ax=ax)
            ax.set(xlabel="Combination Type" if row == 1 else "",
                   xticks=range(len(COMBO_ORDER)),
                   ylim=(0.38, 1.02) if metric == "top1_accuracy" else (0.75, 1.02))
            ax.set_xticklabels(COMBO_LABELS if row == 1 else [""] * len(COMBO_ORDER),
                               rotation=45, ha="right")
            if row == 0:
                ax.set_title(f"Split: {split}", fontweight="bold")
            ax.get_legend().remove()
        axes[row, 0].set_ylabel(ylabel)

    handles = [mpatches.Patch(color=c, label=e) for e, c in PALETTE.items()]
    fig.legend(handles=handles, title="Embedding Type",
               loc="lower center", bbox_to_anchor=(0.5, -0.12),
               frameon=True, ncol=3)
    fig.suptitle("Top-1 and Top-5 Accuracy Distribution by Combination Type",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.15, bottom=0.15)
    fig.savefig(out_path, dpi=dpi)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", required=True, help="Path to summary_metrics_slim.tsv")
    p.add_argument("--out",   "-o", default=None,  help="Output path (default: fig2_strip_plot.svg)")
    p.add_argument("--dpi",         default=300, type=int, help="Resolution (default: 300)")
    args = p.parse_args()
    out = Path(args.out) if args.out else Path(args.input).parent / "fig2_strip_plot.svg"
    main(args.input, out, args.dpi)
