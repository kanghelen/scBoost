import argparse, joblib, sys, os
import scanpy as sc
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report

parser = argparse.ArgumentParser(description="Test a pre-trained stacking ensemble on new/held-out data")
parser.add_argument("--input", required=True, help="Path to input .h5ad file")
parser.add_argument("--model", required=True, help="Path to saved stacking_ensemble .pkl file")
parser.add_argument("--sample_name", required=True, help="Sample name for output filenames")
parser.add_argument("--split_type", default="stratified",
                    help="'stratified' or an adata.obs column name (e.g. donor_id, batch)")
parser.add_argument("--holdout_value", default=None, help="Value to hold out as test set")
parser.add_argument("--test_size", type=float, default=0.1, help="Test fraction (stratified only)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--out_dir", default="/data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs",
                    help="Output directory for results")
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stacking_adata import StackingEnsemble
from split_stratified import stratified_split

# Load trained model & data
stacker = joblib.load(args.model)
print(f"Loaded model from {args.model}")
print(f"Model methods: {stacker.obsm_keys}")

adata = sc.read_h5ad(args.input)

# Validate that all required embeddings are present
available = set(adata.obsm.keys())
required = set(stacker.obsm_keys)
missing = required - available
if missing:
    print(f"ERROR: input data is missing embeddings required by the model.")
    print(f"  Model requires: {sorted(required)}")
    print(f"  Data has:       {sorted(available)}")
    print(f"  Missing:        {sorted(missing)}")
    sys.exit(1)
print(f"Using embeddings: {stacker.obsm_keys}")

# Split train/test
if args.split_type != "stratified" and args.split_type in adata.obs.columns:
    col = args.split_type
    vals = adata.obs[col].astype(str).unique()
    if args.holdout_value in vals:
        val = args.holdout_value
    else:
        counts = adata.obs[col].astype(str).value_counts()
        target = args.test_size * adata.n_obs
        val = counts.iloc[(counts - target).abs().argsort()].index[0]
    adata.obs["split"] = np.where(adata.obs[col].astype(str) == val, "test", "train")
    print(f"Holdout: {col}='{val}' → {(adata.obs['split']=='train').sum()} train, "
          f"{(adata.obs['split']=='test').sum()} test")
else:
    stratified_split(adata, label_key="cell_type", batch_key="batch",
                     test_size=args.test_size, random_state=args.seed)

train = adata[adata.obs["split"] == "train"]
test  = adata[adata.obs["split"] == "test"]

# Remove test cells with cell types unseen during training
train_types = set(stacker.label_encoder_.classes_)
test_types = set(test.obs["cell_type"].unique())
unseen = test_types - train_types
if unseen:
    mask = test.obs["cell_type"].isin(train_types)
    n_removed = (~mask).sum()
    print(f"Removing {n_removed} test cells with unseen cell types: {sorted(unseen)}")
    test = test[mask]

# Predict & evaluate
stacker.predict(test, key_added="stacking_pred")
y_true = np.asarray(test.obs["cell_type"])
y_pred = np.asarray(test.obs["stacking_pred"])

proba = stacker.predict_proba(test) # get the full probability matrix (n_cells x n_classes)
y_true_enc = stacker.label_encoder_.transform(y_true) # convert the true labels to integers
k = min(5, proba.shape[1]) # handles cases where you have fewer than 5 cell types
top_k_indices = np.argsort(proba, axis=1)[:, -k:] # get the indices of the top k probabilities for each cell
top5_acc = np.mean(np.any(top_k_indices == y_true_enc[:, None], axis=1))

report_labels = sorted(set(y_true))
top5_hit = np.any(top_k_indices == y_true_enc[:, None], axis=1)
per_ct_top5 = {ct: top5_hit[y_true == ct].mean() for ct in report_labels}
macro_top5 = np.mean(list(per_ct_top5.values()))
print(f"\nTop-1 Accuracy:        {accuracy_score(y_true, y_pred):.4f}")
print(f"Top-5 Accuracy:        {top5_acc:.4f}  (k={k})")
print(f"Top-5 Accuracy (macro):{macro_top5:.4f}")
print(f"Macro F1:              {f1_score(y_true, y_pred, average='macro', labels=report_labels):.4f}")
print(f"Weighted F1:           {f1_score(y_true, y_pred, average='weighted', labels=report_labels):.4f}")
print(f"({len(report_labels)} cell types with support > 0)")
report_df = pd.DataFrame(classification_report(y_true, y_pred, labels=report_labels, digits=3, zero_division=0, output_dict=True)).T
report_df.insert(0, "top5_accuracy", report_df.index.map(lambda x: per_ct_top5.get(x, "")))
if "macro avg" in report_df.index:
    report_df.at["macro avg", "top5_accuracy"] = macro_top5
if "weighted avg" in report_df.index:
    report_df.at["weighted avg", "top5_accuracy"] = top5_acc
train_counts = getattr(stacker, "train_support_", None)
if train_counts is None:
    train_counts = train.obs["cell_type"].value_counts().to_dict()
report_df.insert(0, "train_support", report_df.index.map(lambda x: train_counts.get(x, 0)))
print("\nPer-cell-type classification report:")
print(report_df.to_string())
print("\nMethod importance:", stacker.get_method_importance())

# Save
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
if args.split_type != "stratified" and args.split_type in adata.obs.columns:
    split_tag = f"holdout_{col}_{val}"
else:
    split_tag = f"stratified_{args.test_size}"
tag = f"{args.sample_name}_{split_tag}_retest"
report_df.to_csv(os.path.join(out_dir, f"{tag}_classification_report.tsv"), sep="\t", index_label="cell_type")
print(f"\nSaved to {out_dir}/{tag}_classification_report.tsv")
