import argparse, os, sys, joblib
import scanpy as sc
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description="Concatenation + XGBoost pipeline for scRNA-seq embeddings")
parser.add_argument("--input", required=True, help="Path to input .h5ad file")
parser.add_argument("--sample_name", required=True, help="Sample name for output filenames")
parser.add_argument("--methods", nargs="+", required=True,
                    help="obsm keys for embedding methods to concatenate (2+ methods)")
parser.add_argument("--split_type", default="stratified",
                    help="'stratified' for random split, or an adata.obs column name "
                         "(e.g. donor, batch) to hold out one value from that column")
parser.add_argument("--holdout_value", default=None, help="Value to hold out (random if omitted)")
parser.add_argument("--test_size", type=float, default=0.1, help="Test fraction (stratified only)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--gpu", type=int, default=None, help="GPU device ID (e.g. 0). CPU if omitted.")
parser.add_argument("--no_batch_feature", action="store_true",
                    help="Disable batch one-hot encoding in classifier")
parser.add_argument("--out_dir", default="/data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs",
                    help="Output directory for results")
args = parser.parse_args()

if args.gpu is not None:
    import torch
    torch.cuda.set_device(args.gpu)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concat_xgboost_adata import ConcatXGBoost
from split_stratified import stratified_split
from eval_report import build_classification_report

## Load Data
device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
adata = sc.read_h5ad(args.input)
methods = args.methods
combo_name = "_".join(sorted(methods))

print(f"=== Concat+XGBoost: {combo_name} ===")
print(f"  Methods: {methods}")
print(f"  Dims: {[adata.obsm[k].shape[1] for k in methods]}")
print(f"  Total concat dim: {sum(adata.obsm[k].shape[1] for k in methods)}")

# Split train/test (identical to example_combined_calls.py)
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

# Remove test cells whose cell_type was never seen in training
train_types = set(train.obs["cell_type"].unique())
test_types = set(test.obs["cell_type"].unique())
unseen = test_types - train_types
if unseen:
    mask = test.obs["cell_type"].isin(train_types)
    n_removed = (~mask).sum()
    print(f"Removing {n_removed} test cells with unseen cell types: {sorted(unseen)}")
    test = test[mask]

# Train & predict
clf = ConcatXGBoost(obsm_keys=methods, label_key="cell_type", batch_key="batch",
                    use_batch_feature=not args.no_batch_feature, device=device)
clf.fit(train)
clf.predict(test, key_added="concat_pred")

# --- Evaluate on test set ---
y_true = np.asarray(test.obs["cell_type"])
y_pred = np.asarray(test.obs["concat_pred"])
proba = clf.predict_proba(test)
report_df = build_classification_report(y_true, y_pred, proba, clf.label_encoder_, train)

# Method importance
print("\nMethod importance (feature importance by method):")
print(clf.get_method_importance())

# --- Save results ---
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
if args.split_type != "stratified" and args.split_type in adata.obs.columns:
    split_tag = f"holdout_{col}_{val}"
else:
    split_tag = f"stratified_{args.test_size}"
tag = f"{args.sample_name}_concat_{combo_name}_{split_tag}"
report_df.to_csv(os.path.join(out_dir, f"{tag}_classification_report.tsv"), sep="\t", index_label="cell_type")
pd.DataFrame(clf.get_method_importance().items(), columns=["method", "importance"]).to_csv(
    os.path.join(out_dir, f"{tag}_method_importance.tsv"), sep="\t", index=False)
ts = datetime.now().strftime("%y%m%d_%H%M%S")
joblib.dump(clf, os.path.join(out_dir, f"{ts}_{tag}_concat_xgboost.pkl"))
print(f"\nSaved to {out_dir}/")
