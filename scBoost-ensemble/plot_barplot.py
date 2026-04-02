"""
Plot Top-1 and Top-5 accuracy horizontal bar plots (top 10 methods per panel).

Usage:
    python plot_barplot.py --input summary_metrics_slim.tsv
    python plot_barplot.py --input summary_metrics_slim.tsv --out fig.svg --dpi 600
"""
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
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

TYPE_COLORS = {
    "single":       "#888780",
    "concat":       "#38b4f7",
    "stacked":      "#EF9F27",
    "stacked_perm": "#0141a8",
}

TYPE_LABELS = {
    "single":       "Single method",
    "concat":       "Concatenation",
    "stacked_perm": "scBoost-ensemble (3-method)",
    "stacked":      "scBoost-ensemble (all methods)",
}

METRICS = [("top1_accuracy", "Top-1 Accuracy"), ("top5_accuracy", "Top-5 Accuracy")]
N_TOP = 10


def make_label(combo, methods):
    m = methods.replace("_", " + ")
    if combo == "single":
        return m
    elif combo == "concat":
        return f"{m} (concat)"
    elif combo == "stacked":
        return f"All methods (scBoost-ensemble)"
    elif combo == "stacked_perm":
        return f"{m} (scBoost-ensemble)"
    return m


def main(input_path, out_path, dpi=300):
    df = pd.read_csv(input_path, sep="\t").rename(columns={
        "Combination/Model Type": "combo", "Embedding Type": "embedding",
        "Methods": "methods", "Split": "split"})
    df = df.dropna(subset=["top1_accuracy", "top5_accuracy"])
    df = df.drop_duplicates(subset=["combo", "embedding", "methods", "split"])

    plt.rcParams.update(FONT_RC)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 4.5))
    fig.subplots_adjust(hspace=0.55, wspace=0.85, left=0.22, right=0.93,
                        top=0.91, bottom=0.10)

    panels = [
        ("stratified", "top1_accuracy", "Stratified split — Top-1 accuracy"),
        ("stratified", "top5_accuracy", "Stratified split — Top-5 accuracy"),
        ("holdout",    "top1_accuracy", "Donor hold-out — Top-1 accuracy"),
        ("holdout",    "top5_accuracy", "Donor hold-out — Top-5 accuracy"),
    ]

    for ax, (split, metric, title) in zip(axes.flat, panels):
        sub = df[df["split"] == split].copy()
        sub = sub.sort_values(metric, ascending=False).head(N_TOP)
        sub = sub.iloc[::-1]  # reverse so best is at top

        labels = [make_label(r["combo"], r["methods"]) for _, r in sub.iterrows()]
        values = sub[metric].values
        colors = [TYPE_COLORS.get(r["combo"], "#aaa") for _, r in sub.iterrows()]

        y_pos = range(len(sub))
        ax.barh(y_pos, values, color=colors, edgecolor="white",
                linewidth=0.3, height=0.7)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(title, fontsize=7, fontweight="bold", pad=5)

        vmin, vmax = values.min(), values.max()
        span = vmax - vmin
        ax.set_xlim(vmin - span * 0.15 - 0.002, vmax + span * 0.55)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        for i, val in enumerate(values):
            ax.text(val + span * 0.03, i, f"{val:.4f}",
                    va="center", ha="left", fontsize=6, color="#444")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.2, linewidth=0.3)

    handles = [mpatches.Patch(color=c, label=l)
               for key, l in TYPE_LABELS.items()
               for c in [TYPE_COLORS[key]]]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.58, -0.02), frameon=True, ncol=2,
               edgecolor="#ccc", fontsize=6)

    fig.savefig(out_path, dpi=dpi)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", required=True, help="Path to summary_metrics_slim.tsv")
    p.add_argument("--out",   "-o", default=None,  help="Output path (default: fig_top10_barplot.svg)")
    p.add_argument("--dpi",         default=300, type=int, help="Resolution (default: 300)")
    args = p.parse_args()
    out = Path(args.out) if args.out else Path(args.input).parent / "fig_top10_barplot.svg"
    main(args.input, out, args.dpi)
