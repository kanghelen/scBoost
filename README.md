# scBoost

**scBoost-ensemble for cell-type classification from multiple scRNA-seq embeddings.**

scBoost-ensemble trains per-method XGBoost base learners on each embedding view stored in an AnnData object (`adata.obsm`), then combines their out-of-fold probability predictions through a meta-learner.
This stacking strategy lets the ensemble leverage the complementary strengths of different integration and embedding methods (e.g., Harmony, Scanorama, scVI, scGPT, scANVI).

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│ AnnData (.h5ad)                                             │
│   obsm: X_harmony, X_scanorama, X_scvi, ...                 │
│   obs:  cell_type, batch, [donor_id, ...]                   │
└──────────────┬──────────────────────────────────────────────┘
               │
        ┌──────▼──────┐
        │  Split data │  stratified or user-defined holdout set (e.g. donor)
        └──────┬──────┘
               │
    ┌──────────▼──────────┐
    │  Base learners (CV) │  one XGBoost per embedding method,
    │  out-of-fold probs  │  trained in parallel across GPUs
    └──────────┬──────────┘
               │
       ┌───────▼───────┐
       │  Meta-learner │  XGBoost on stacked probabilities
       └───────┬───────┘
               │
       ┌───────▼───────┐
       │  Predictions  │  cell-type labels + probabilities
       │  + evaluation │  per-cell-type report, method importance
       └───────────────┘
```

## Installation

### Dependencies

- Python >= 3.9
- [anndata](https://anndata.readthedocs.io) / [scanpy](https://scanpy.readthedocs.io)
- [xgboost](https://xgboost.readthedocs.io) (>= 2.0 recommended for `device="cuda:N"` support)
- scikit-learn
- numpy, pandas, joblib
- matplotlib, seaborn (plotting only)

```bash
pip install anndata scanpy xgboost scikit-learn numpy pandas joblib matplotlib seaborn
```

## Quick start

### Input format

scBoost operates on **AnnData** `.h5ad` files with:

- **`adata.obsm`** — one or more precomputed embedding matrices (e.g. `X_harmony`, `X_scanorama`, `X_scvi`)
- **`adata.obs["cell_type"]`** — cell-type labels
- **`adata.obs["batch"]`** — batch identifiers

### Train a stacking ensemble

```bash
python scBoost-ensemble/example_combined_calls.py \
    --input data/SketchProcessed_10k_geosketch.h5ad \
    --sample_name TabulaSapiens_10k \
    --methods X_harmony X_scanorama X_scvi \
    --split_type stratified --test_size 0.1 --seed 42 \
    --gpu 0 --n_gpus 2 \
    --out_dir outputs/
```

### Train a single-method baseline

```bash
python scBoost-ensemble/example_single_method.py \
    --input data/SketchProcessed_10k_geosketch.h5ad \
    --sample_name my_dataset \
    --method X_harmony \
    --split_type stratified --test_size 0.1 --seed 42 \
    --gpu 0 \
    --out_dir outputs/
```

### Evaluate a saved model on new data

```bash
python scBoost-ensemble/test_only.py \
    --input data/new_data.h5ad \
    --model outputs/stacking_ensemble.pkl \
    --sample_name my_dataset \
    --split_type donor_id --holdout_value TSP7 \
    --out_dir outputs/
```
The above example evaluates on cells with donor_id labeled as "TSP7".

### Hold out an entire donor / batch for testing

Use `--split_type` with an `adata.obs` column name instead of `stratified`:

```bash
python scBoost-ensemble/example_combined_calls.py \
    --input data.h5ad \
    --sample_name holdout_experiment \
    --methods X_harmony X_scanorama X_scvi \
    --split_type donor_id --test_size 0.1 \
    --no_batch_feature \
    --out_dir outputs/
```

## CLI arguments

| Argument | Description | Default |
|---|---|---|
| `--input` | Path to `.h5ad` file | required |
| `--sample_name` | Tag for output filenames | required |
| `--methods` | `obsm` keys to use as views | required |
| `--split_type` | `stratified` or an `obs` column name (e.g. `donor_id`) | `stratified` |
| `--holdout_value` | Specific value to hold out as test set | required if `split_type` is an `obs` column name |
| `--test_size` | Test fraction (stratified splits only) | `0.1` |
| `--seed` | Random seed | `42` |
| `--gpu` | Starting GPU device ID; CPU if omitted | `None` |
| `--n_gpus` | Number of GPUs for parallel base-learner training | `1` |
| `--no_batch_feature` | Disable batch one-hot in meta-learner | `False` |
| `--out_dir` | Output directory | `outputs/` |

## Outputs

Each run produces:

| File | Contents |
|---|---|
| `*_classification_report.tsv` | Per-cell-type precision, recall, F1, top-5 accuracy, and support |
| `*_method_importance.tsv` | Relative importance of each embedding method in the meta-learner |
| `*_split.tsv` | Train/test assignment for each cell barcode |
| `*_stacking_ensemble.pkl` | Serialised model (loadable with `joblib.load`) |


## Batch sweep example

Run 3-method combinations from a set of 7 embeddings, distributing across GPUs:

```bash
METHODS=(Geoksketch Harmony NMF Python_PCA Scanorama scANVI scVI)
for ((i=0; i<${#METHODS[@]}-2; i++)); do
  for ((j=i+1; j<${#METHODS[@]}-1; j++)); do
    for ((k=j+1; k<${#METHODS[@]}; k++)); do
      python scBoost-ensemble/example_combined_calls.py \
          --input data.h5ad \
          --sample_name "combo_${METHODS[$i]}_${METHODS[$j]}_${METHODS[$k]}" \
          --methods ${METHODS[$i]} ${METHODS[$j]} ${METHODS[$k]} \
          --split_type donor_id --gpu $((job++ % 8)) --no_batch_feature &
    done
  done
done
wait
```

Aggregate results afterward:

```bash
python scBoost-ensemble/extract_summary_metrics.py --outputs_dir outputs/
python scBoost-ensemble/plot_barplot.py --input outputs/summary_metrics_slim.tsv --out figures/barplot.svg
```
