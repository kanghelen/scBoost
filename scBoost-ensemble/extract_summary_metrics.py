"""
Extract weighted avg, micro avg (accuracy), and macro avg summary rows
from all *_classification_report.tsv files under outputs/.
Produces a single summary TSV with one row per experiment.
"""

import argparse
from pathlib import Path
import pandas as pd

DIR_MAP = {
    ".": "stacked_ensemble",
    "permutations": "stacked_ensemble_permutations",
}

# Maps source_dir to short combination type label
COMBO_MAP = {
    "stacked_ensemble": "stacked",
    "concat_pairs": "concat",
    "single_method": "single",
    "stacked_ensemble_permutations": "stacked_perm",
}


def clean_methods(s: str) -> str:
    """Apply all method name normalizations."""
    s = s.replace("Geoksketch", "Geosketch")
    s = s.replace("Python_PCA", "PCA")
    s = s.replace("_PCA100d", "").replace("PCA100d_", "")
    return s


def shorten_experiment(experiment: str, source_dir: str) -> tuple[str, str, str, str, str]:
    """
    Returns (short_experiment, combo_type, embedding_type, methods, split).
    Short format: {combo}_{embedding}_{methods}_{split}
    """
    import re

    combo = COMBO_MAP.get(source_dir, source_dir)

    # Determine embedding type from original experiment name
    if "SketchProcessed" in experiment:
        embedding = "classic"
    elif "scFM_full" in experiment or "with_scFM_full" in experiment:
        embedding = "scFM_full"
    else:
        embedding = "scFM"

    # Determine split
    if "holdout" in experiment:
        split = "holdout"
    else:
        split = "stratified"

    # Extract methods based on source type
    exp = experiment
    exp = re.sub(r"_stratified_0\.1$", "", exp)
    exp = re.sub(r"_holdout_donor_id_TSP7$", "", exp)

    if source_dir == "stacked_ensemble":
        methods = "All"
    elif source_dir == "concat_pairs":
        m = re.search(r"concat2_(?:holdout_)?(.+?)_concat_", exp)
        methods = m.group(1) if m else exp
    elif source_dir == "single_method":
        m = re.search(r"_single_(.+)$", exp)
        methods = m.group(1) if m else exp
    elif source_dir == "stacked_ensemble_permutations":
        m = re.search(r"combo\d+_(.+)$", exp)
        methods = m.group(1) if m else exp
    else:
        methods = exp

    methods = clean_methods(methods)
    short_exp = f"{combo}_{embedding}_{methods}_{split}"
    return short_exp, combo, embedding, methods, split


def parse_report(path: Path, outputs_root: Path) -> dict:
    df = pd.read_csv(path, sep="\t", index_col="cell_type")

    rel = path.relative_to(outputs_root)
    subdir = str(rel.parent)
    source_dir = DIR_MAP.get(subdir, subdir)

    filename = path.name.replace("_classification_report.tsv", "")
    experiment = f"{source_dir}_{filename}"

    if "holdout" in filename:
        data_split_type = "holdout"
    else:
        data_split_type = "stratified"

    short_exp, combo, embedding, methods, split = shorten_experiment(experiment, source_dir)

    row = {
        "experiment": short_exp,
        "Combination/Model Type": combo,
        "Embedding Type": embedding,
        "Methods": methods,
        "Split": split,
        "source_dir": source_dir,
        "data_split_type": data_split_type,
    }

    # accuracy row = micro avg (precision == recall == f1 == top-1 accuracy)
    if "accuracy" in df.index:
        acc = df.loc["accuracy"]
        row["top1_accuracy"] = acc["precision"]
        row["micro_precision"] = acc["precision"]
        row["micro_recall"] = acc["recall"]
        row["micro_f1"] = acc["f1-score"]
        row["support"] = acc["support"]

    if "macro avg" in df.index:
        m = df.loc["macro avg"]
        row["macro_top5_accuracy"] = m["top5_accuracy"]
        row["macro_precision"] = m["precision"]
        row["macro_recall"] = m["recall"]
        row["macro_f1"] = m["f1-score"]

    if "weighted avg" in df.index:
        w = df.loc["weighted avg"]
        row["top5_accuracy"] = w["top5_accuracy"]
        row["weighted_top5_accuracy"] = w["top5_accuracy"]
        row["weighted_precision"] = w["precision"]
        row["weighted_recall"] = w["recall"]
        row["weighted_f1"] = w["f1-score"]

    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs_dir",
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Root outputs directory to scan (default: outputs/ next to this script)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output TSV path (default: <outputs_dir>/summary_metrics.tsv)",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_dir)
    reports = sorted(outputs_root.glob("**/*_classification_report.tsv"))
    print(f"Found {len(reports)} classification report(s) under {outputs_root}")

    rows = [parse_report(p, outputs_root) for p in reports]
    summary = pd.DataFrame(rows)

    col_order = [
        "experiment",
        "Combination/Model Type",
        "Embedding Type",
        "Methods",
        "Split",
        "source_dir",
        "data_split_type",
        "support",
        "top1_accuracy",
        "top5_accuracy",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_top5_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_top5_accuracy",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]
    summary = summary[[c for c in col_order if c in summary.columns]]

    out_path = Path(args.out) if args.out else outputs_root / "summary_metrics.tsv"
    summary.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(summary)} rows to {out_path}")

    # Write slim summary with key columns only
    slim_cols = [
        "Combination/Model Type",
        "Embedding Type",
        "Methods",
        "Split",
        "top1_accuracy",
        "top5_accuracy",
    ]
    slim = summary[[c for c in slim_cols if c in summary.columns]]
    slim_path = out_path.parent / (out_path.stem + "_slim.tsv")
    slim.to_csv(slim_path, sep="\t", index=False)
    print(f"Wrote slim summary ({len(slim_cols)} columns) to {slim_path}")


if __name__ == "__main__":
    main()
