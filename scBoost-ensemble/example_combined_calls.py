import argparse, os, sys, joblib
import scanpy as sc
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description="Ensemble stacking pipeline for scRNA-seq embeddings")
parser.add_argument("--input", required=True, help="Path to input .h5ad file")
parser.add_argument("--sample_name", required=True, help="Sample name for output filenames")
parser.add_argument("--methods", nargs="+", required=True, help="obsm keys for embedding methods")
parser.add_argument("--split_type", default="stratified",
                    help="'stratified' for random split, or an adata.obs column name "
                         "(e.g. donor, batch) to hold out one value from that column")
parser.add_argument("--holdout_value", default=None, help="Value to hold out (random if omitted)")
parser.add_argument("--test_size", type=float, default=0.1, help="Test fraction (stratified only)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--gpu", type=int, default=None, help="Starting GPU device ID (e.g. 0). CPU if omitted.")
parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs for parallel method training (default 1)")
parser.add_argument("--no_batch_feature", action="store_true",
                    help="Disable batch one-hot encoding in meta-learner (useful for holdout splits)")
parser.add_argument("--out_dir", default="/data/cb/helenk/scRNA-seq_Methods_Boosting/260312_ensemble_stacking/outputs",
                    help="Output directory for results")
args = parser.parse_args()

if args.gpu is not None:
    import torch
    torch.cuda.set_device(args.gpu)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consensus_adata import ConsensusGraphEmbedding
from contrastive_adata import CrossBatchSupCon
from stacking_adata import StackingEnsemble
from split_stratified import stratified_split
from eval_report import build_classification_report

# Your AnnData with embeddings already computed
# adata.obsm: "X_harmony", "X_scanorama", "X_scvi", ...
# adata.obs:  "cell_type", "batch"

## Load Data
device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
adata = sc.read_h5ad(args.input)
methods = args.methods

# # --- Stage 1: Consensus graph → adata.obsm["X_consensus_graph"] ---
# cge = ConsensusGraphEmbedding(n_neighbors=30, strategy="jaccard", device=device)
# cge.fit_transform(adata, obsm_keys=methods, batch_key="batch",
#                   key_added="X_consensus_graph")

# # --- Stage 2: Contrastive refinement → adata.obsm["X_supcon"] ---
# input_dim = sum(adata.obsm[k].shape[1] for k in methods)
# supcon = CrossBatchSupCon(input_dim=input_dim)
# supcon.fit_transform(adata, obsm_keys=methods, label_key="cell_type",
#                      batch_key="batch", key_added="X_supcon", epochs=10000)

# --- Stage 3: Stacking ensemble over all views ---
# all_views = methods + ["X_consensus_graph", "X_supcon"]
all_views = methods

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
stacker = StackingEnsemble(obsm_keys=all_views, label_key="cell_type", batch_key="batch",
                           use_batch_feature=not args.no_batch_feature,
                           device=device, n_gpus=args.n_gpus)
stacker.fit(train)
stacker.predict(test, key_added="stacking_pred")  # writes to test.obs

# --- Evaluate on test set ---
y_true = np.asarray(test.obs["cell_type"])
y_pred = np.asarray(test.obs["stacking_pred"])
proba = stacker.predict_proba(test)
report_df = build_classification_report(y_true, y_pred, proba, stacker.label_encoder_, train)

# Inspect which methods the meta-learner relies on
print("\nMethod importance:")
print(stacker.get_method_importance())

# --- Save results ---
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
if args.split_type != "stratified" and args.split_type in adata.obs.columns:
    split_tag = f"holdout_{col}_{val}"
else:
    split_tag = f"stratified_{args.test_size}"
tag = f"{args.sample_name}_{split_tag}"
report_df.to_csv(os.path.join(out_dir, f"{tag}_classification_report.tsv"), sep="\t", index_label="cell_type")
pd.DataFrame(stacker.get_method_importance().items(), columns=["method", "importance"]).to_csv(
    os.path.join(out_dir, f"{tag}_method_importance.tsv"), sep="\t", index=False)
adata.obs[["split"]].to_csv(os.path.join(out_dir, f"{tag}_split.tsv"), sep="\t", index_label="cell_barcode")
ts = datetime.now().strftime("%y%m%d_%H%M%S")
joblib.dump(stacker, os.path.join(out_dir, f"{ts}_{tag}_stacking_ensemble.pkl"))
print(f"\nSaved to {out_dir}/")
