"""
Technique 7: Stacking Ensemble — AnnData interface.
Reads embeddings from adata.obsm, labels/batch from adata.obs.
Supports multi-GPU parallel training across methods.
"""

import time
import numpy as np
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import StratifiedKFold
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


class StackingEnsemble:
    def __init__(
        self,
        obsm_keys: List[str],
        label_key: str = "cell_type",
        batch_key: str = "batch",
        use_batch_feature: bool = True,
        n_folds: int = 5,
        device: str = "cpu",
        n_gpus: int = 1,
        base_params: Optional[dict] = None,
        meta_params: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        device : str
            "cpu" or "cuda:N" (GPU starting ID for XGBoost).
        n_gpus : int
            Number of GPUs to use. Methods are distributed round-robin
            across cuda:gpu_start .. cuda:gpu_start+n_gpus-1.
        use_batch_feature : bool
            If True, append one-hot batch IDs to meta-learner features.
            Set False to avoid issues with unseen batches at test time.
        """
        self.obsm_keys = obsm_keys
        self.label_key = label_key
        self.batch_key = batch_key
        self.use_batch_feature = use_batch_feature
        self.n_folds = n_folds
        self.n_gpus = n_gpus
        self.n_cell_types = None

        if device.startswith("cuda"):
            parts = device.split(":")
            self.gpu_start = int(parts[1]) if len(parts) > 1 else 0
        else:
            self.gpu_start = 0
        self.device = device

        gpu_kw = _xgb_gpu_kw(device)
        print(f"XGBoost {xgb.__version__}, device={device}, n_gpus={n_gpus}, params={gpu_kw}")

        self.base_params = base_params or dict(
            objective="multi:softprob", max_depth=6, learning_rate=0.1,
            n_estimators=200, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, eval_metric="mlogloss", verbosity=0,
            **gpu_kw,
        )
        self.meta_params = meta_params or dict(
            objective="multi:softprob", max_depth=4, learning_rate=0.05,
            n_estimators=300, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="mlogloss", verbosity=0,
            **gpu_kw,
        )
        self.base_classifiers_ = {}
        self.meta_classifier_ = None
        self.label_encoder_ = LabelEncoder()

    def _extract(self, adata: AnnData):
        embs = {k: np.asarray(adata.obsm[k]) for k in self.obsm_keys}
        labels = np.asarray(adata.obs[self.label_key])
        batch = np.asarray(adata.obs[self.batch_key])
        return embs, labels, batch

    def _train_one_method(self, key, X, labels, folds, gpu_id):
        """Train all folds for one method on one GPU. Thread-safe."""
        C = self.n_cell_types
        n = len(labels)
        meta_col = np.zeros((n, C))
        classifiers = []
        device_str = f"cuda:{gpu_id}" if self.device != "cpu" else "cpu"
        gpu_kw = _xgb_gpu_kw(device_str)

        for fi, (tr, va) in enumerate(folds):
            p = {**self.base_params, "num_class": C, **gpu_kw}
            clf = xgb.XGBClassifier(**p)
            clf.fit(X[tr], labels[tr],
                    eval_set=[(X[va], labels[va])], verbose=False)
            meta_col[va] = clf.predict_proba(X[va])
            classifiers.append(clf)
            print(f"  Base: {key} fold {fi+1}/{self.n_folds} [GPU {gpu_id}] done")
        return key, classifiers, meta_col

    def _oof(self, embs, labels):
        n, C = len(labels), self.n_cell_types
        meta = np.zeros((n, len(self.obsm_keys) * C))
        skf = StratifiedKFold(self.n_folds, shuffle=True, random_state=42)
        folds = list(skf.split(embs[self.obsm_keys[0]], labels))

        use_parallel = self.device != "cpu" and self.n_gpus > 1

        if use_parallel:
            print(f"  Parallel training: {len(self.obsm_keys)} methods across "
                  f"{self.n_gpus} GPUs ({self.gpu_start}..{self.gpu_start + self.n_gpus - 1})")
            t_parallel = time.time()
            with ThreadPoolExecutor(max_workers=self.n_gpus) as pool:
                futures = {}
                submit_times = {}
                for mi, key in enumerate(self.obsm_keys):
                    gpu_id = self.gpu_start + (mi % self.n_gpus)
                    f = pool.submit(self._train_one_method, key, embs[key],
                                    labels, folds, gpu_id)
                    futures[f] = mi
                    submit_times[mi] = time.time()

                for f in as_completed(futures):
                    mi = futures[f]
                    key, classifiers, meta_col = f.result()
                    cs, ce = mi * C, (mi + 1) * C
                    meta[:, cs:ce] = meta_col
                    self.base_classifiers_[key] = classifiers
                    elapsed = time.time() - submit_times[mi]
                    print(f"  Base: {key} complete ({elapsed:.1f}s) "
                          f"[GPU {self.gpu_start + (mi % self.n_gpus)}]")
            print(f"  All base classifiers done ({time.time()-t_parallel:.1f}s wall time)")
        else:
            for mi, key in enumerate(self.obsm_keys):
                t0 = time.time()
                gpu_id = self.gpu_start
                key, classifiers, meta_col = self._train_one_method(
                    key, embs[key], labels, folds, gpu_id)
                cs, ce = mi * C, (mi + 1) * C
                meta[:, cs:ce] = meta_col
                self.base_classifiers_[key] = classifiers
                print(f"  Base: {key} complete ({time.time()-t0:.1f}s)")
        return meta

    def _test_meta(self, embs):
        n, C = next(iter(embs.values())).shape[0], self.n_cell_types
        meta = np.zeros((n, len(self.obsm_keys) * C))
        for mi, key in enumerate(self.obsm_keys):
            cs, ce = mi * C, (mi + 1) * C
            probs = [c.predict_proba(embs[key]) for c in self.base_classifiers_[key]]
            meta[:, cs:ce] = np.mean(probs, axis=0)
        return meta

    def fit(self, adata: AnnData):
        t_start = time.time()
        embs, labels, batch = self._extract(adata)
        enc_labels = self.label_encoder_.fit_transform(labels)
        self.n_cell_types = len(self.label_encoder_.classes_)
        unique, counts = np.unique(labels, return_counts=True)
        self.train_support_ = dict(zip(unique, counts.astype(int)))
        print(f"Stacking fit: {adata.n_obs} cells, {self.n_cell_types} cell types, "
              f"{len(self.obsm_keys)} methods")

        print("Training base classifiers (out-of-fold)...")
        meta = self._oof(embs, enc_labels)

        if getattr(self, "use_batch_feature", True):
            self._batch_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(batch.reshape(-1, 1))
            meta = np.hstack([meta, self._batch_enc.transform(batch.reshape(-1, 1))])

        print(f"Training meta-learner on {meta.shape[1]} features...")
        p = {**self.meta_params, "num_class": self.n_cell_types}
        self.meta_classifier_ = xgb.XGBClassifier(**p)
        self.meta_classifier_.fit(meta, enc_labels, verbose=False)
        print(f"Stacking fit complete ({time.time()-t_start:.1f}s total)")
        return self

    def predict(self, adata: AnnData, key_added: str = "stacking_pred") -> np.ndarray:
        print(f"Predicting on {adata.n_obs} cells...")
        t0 = time.time()
        embs, _, batch = self._extract(adata)
        meta = self._test_meta(embs)
        if getattr(self, "use_batch_feature", True):
            meta = np.hstack([meta, self._batch_enc.transform(batch.reshape(-1, 1))])
        preds = self.label_encoder_.inverse_transform(self.meta_classifier_.predict(meta))
        adata.obs[key_added] = preds
        print(f"Prediction complete ({time.time()-t0:.1f}s)")
        return preds

    def predict_proba(self, adata: AnnData) -> np.ndarray:
        embs, _, batch = self._extract(adata)
        meta = self._test_meta(embs)
        if getattr(self, "use_batch_feature", True):
            meta = np.hstack([meta, self._batch_enc.transform(batch.reshape(-1, 1))])
        return self.meta_classifier_.predict_proba(meta)

    def get_method_importance(self) -> Dict[str, float]:
        imp = self.meta_classifier_.feature_importances_
        C = self.n_cell_types
        scores = {k: imp[i*C:(i+1)*C].sum() for i, k in enumerate(self.obsm_keys)}
        total = sum(scores.values())
        return {k: v/total for k, v in sorted(scores.items(), key=lambda x: -x[1])} if total else scores
