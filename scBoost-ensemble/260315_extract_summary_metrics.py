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

    row = {"experiment": experiment, "source_dir": source_dir, "data_split_type": data_split_type}

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


if __name__ == "__main__":
    main()
