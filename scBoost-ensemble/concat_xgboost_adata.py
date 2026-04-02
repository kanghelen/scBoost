"""
Concatenation + XGBoost baseline — AnnData interface.
Concatenates embeddings from adata.obsm, trains a single XGBoost classifier.
Drop-in comparison against the stacking ensemble approach.
"""

import time
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from anndata import AnnData
from typing import List, Optional, Dict


def _xgb_gpu_kw(device: str) -> dict:
    """Version-safe XGBoost GPU params."""
    use_gpu = device != "cpu"
    xgb_major = int(xgb.__version__.split(".")[0])
    if use_gpu and xgb_major >= 2:
        return dict(tree_method="hist", device=device)
    elif use_gpu:
        return dict(tree_method="gpu_hist")
    else:
        return dict(tree_method="hist")


class ConcatXGBoost:
    def __init__(
        self,
        obsm_keys: List[str],
        label_key: str = "cell_type",
        batch_key: str = "batch",
        use_batch_feature: bool = True,
        device: str = "cpu",
        xgb_params: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        obsm_keys : list of str
            Keys in adata.obsm to concatenate.
        label_key : str
            Column in adata.obs containing cell type labels.
        batch_key : str
            Column in adata.obs containing batch IDs.
        use_batch_feature : bool
            If True, append one-hot batch IDs to concatenated features.
        device : str
            "cpu" or "cuda:N".
        xgb_params : dict, optional
            Override default XGBoost parameters.
        """
        self.obsm_keys = obsm_keys
        self.label_key = label_key
        self.batch_key = batch_key
        self.use_batch_feature = use_batch_feature
        self.device = device
        self.n_cell_types = None

        gpu_kw = _xgb_gpu_kw(device)
        print(f"ConcatXGBoost: XGBoost {xgb.__version__}, device={device}, params={gpu_kw}")

        self.xgb_params = xgb_params or dict(
            objective="multi:softprob", max_depth=6, learning_rate=0.1,
            n_estimators=200, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, eval_metric="mlogloss", verbosity=0,
            **gpu_kw,
        )
        self.classifier_ = None
        self.label_encoder_ = LabelEncoder()

    def _concat(self, adata: AnnData) -> np.ndarray:
        """Concatenate embeddings from obsm_keys into one feature matrix."""
        parts = [np.asarray(adata.obsm[k]) for k in self.obsm_keys]
        return np.hstack(parts)

    def _add_batch(self, X: np.ndarray, batch: np.ndarray, fit: bool = False) -> np.ndarray:
        if not self.use_batch_feature:
            return X
        if fit:
            self._batch_enc = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            ).fit(batch.reshape(-1, 1))
        return np.hstack([X, self._batch_enc.transform(batch.reshape(-1, 1))])

    def fit(self, adata: AnnData):
        t_start = time.time()
        X = self._concat(adata)
        labels = np.asarray(adata.obs[self.label_key])
        batch = np.asarray(adata.obs[self.batch_key])

        enc_labels = self.label_encoder_.fit_transform(labels)
        self.n_cell_types = len(self.label_encoder_.classes_)

        # Track per-method dimension ranges for feature importance
        self._dim_ranges = {}
        offset = 0
        for k in self.obsm_keys:
            d = adata.obsm[k].shape[1]
            self._dim_ranges[k] = (offset, offset + d)
            offset += d

        X = self._add_batch(X, batch, fit=True)

        dims_str = " + ".join(
            f"{k}({adata.obsm[k].shape[1]}d)" for k in self.obsm_keys
        )
        print(f"ConcatXGBoost fit: {adata.n_obs} cells, {self.n_cell_types} types")
        print(f"  Features: {dims_str} = {X.shape[1]} total")

        p = {**self.xgb_params, "num_class": self.n_cell_types}
        self.classifier_ = xgb.XGBClassifier(**p)
        self.classifier_.fit(X, enc_labels, verbose=False)
        print(f"  Training complete ({time.time()-t_start:.1f}s)")
        return self

    def predict(self, adata: AnnData, key_added: str = "concat_pred") -> np.ndarray:
        t0 = time.time()
        X = self._concat(adata)
        batch = np.asarray(adata.obs[self.batch_key])
        X = self._add_batch(X, batch)
        preds = self.label_encoder_.inverse_transform(self.classifier_.predict(X))
        adata.obs[key_added] = preds
        print(f"Prediction: {adata.n_obs} cells ({time.time()-t0:.1f}s)")
        return preds

    def predict_proba(self, adata: AnnData) -> np.ndarray:
        X = self._concat(adata)
        batch = np.asarray(adata.obs[self.batch_key])
        X = self._add_batch(X, batch)
        return self.classifier_.predict_proba(X)

    def get_method_importance(self) -> Dict[str, float]:
        """Aggregate XGBoost feature importances by method."""
        imp = self.classifier_.feature_importances_
        scores = {}
        for k, (start, end) in self._dim_ranges.items():
            scores[k] = imp[start:end].sum()
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        return dict(sorted(scores.items(), key=lambda x: -x[1]))
