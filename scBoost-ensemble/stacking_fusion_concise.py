"""
Technique 7: Stacking Ensemble for scRNA-seq cell type classification.
Trains independent base classifiers on each method's embedding, then a
meta-learner on their out-of-fold probability outputs.
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import f1_score
from typing import Dict, List, Optional


class StackingEnsemble:
    def __init__(
        self,
        method_names: List[str],
        n_cell_types: int,
        n_folds: int = 5,
        base_params: Optional[dict] = None,
        meta_params: Optional[dict] = None,
    ):
        self.method_names = method_names
        self.n_cell_types = n_cell_types
        self.n_folds = n_folds

        self.base_params = base_params or dict(
            objective="multi:softprob", num_class=n_cell_types, max_depth=6,
            learning_rate=0.1, n_estimators=200, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
            eval_metric="mlogloss", verbosity=0,
        )
        self.meta_params = meta_params or dict(
            objective="multi:softprob", num_class=n_cell_types, max_depth=4,
            learning_rate=0.05, n_estimators=300, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1,
            reg_lambda=1.0, tree_method="hist", eval_metric="mlogloss",
            verbosity=0,
        )
        self.base_classifiers_ = {}
        self.meta_classifier_ = None
        self.label_encoder_ = LabelEncoder()

    def _oof_predictions(self, embeddings: Dict[str, np.ndarray], labels: np.ndarray):
        """Generate out-of-fold base predictions (prevents leakage)."""
        n, C = len(labels), self.n_cell_types
        meta = np.zeros((n, len(self.method_names) * C))
        skf = StratifiedKFold(self.n_folds, shuffle=True, random_state=42)

        for mi, name in enumerate(self.method_names):
            X = embeddings[name]
            self.base_classifiers_[name] = []
            cs, ce = mi * C, (mi + 1) * C

            for train_idx, val_idx in skf.split(X, labels):
                clf = xgb.XGBClassifier(**self.base_params)
                clf.fit(X[train_idx], labels[train_idx],
                        eval_set=[(X[val_idx], labels[val_idx])], verbose=False)
                meta[val_idx, cs:ce] = clf.predict_proba(X[val_idx])
                self.base_classifiers_[name].append(clf)
        return meta

    def _test_meta_features(self, embeddings: Dict[str, np.ndarray]):
        """Average fold-model predictions for test data."""
        n, C = next(iter(embeddings.values())).shape[0], self.n_cell_types
        meta = np.zeros((n, len(self.method_names) * C))
        for mi, name in enumerate(self.method_names):
            cs, ce = mi * C, (mi + 1) * C
            probs = [clf.predict_proba(embeddings[name])
                     for clf in self.base_classifiers_[name]]
            meta[:, cs:ce] = np.mean(probs, axis=0)
        return meta

    def fit(self, embeddings: Dict[str, np.ndarray], labels: np.ndarray,
            batch_ids: Optional[np.ndarray] = None):
        enc_labels = self.label_encoder_.fit_transform(labels)
        meta = self._oof_predictions(embeddings, enc_labels)

        if batch_ids is not None:
            self._batch_enc = OneHotEncoder(sparse_output=False).fit(batch_ids.reshape(-1, 1))
            meta = np.hstack([meta, self._batch_enc.transform(batch_ids.reshape(-1, 1))])

        self.meta_classifier_ = xgb.XGBClassifier(**self.meta_params)
        self.meta_classifier_.fit(meta, enc_labels, verbose=False)
        return self

    def predict(self, embeddings: Dict[str, np.ndarray],
                batch_ids: Optional[np.ndarray] = None):
        meta = self._test_meta_features(embeddings)
        if batch_ids is not None and hasattr(self, "_batch_enc"):
            meta = np.hstack([meta, self._batch_enc.transform(batch_ids.reshape(-1, 1))])
        return self.label_encoder_.inverse_transform(self.meta_classifier_.predict(meta))

    def predict_proba(self, embeddings: Dict[str, np.ndarray],
                      batch_ids: Optional[np.ndarray] = None):
        meta = self._test_meta_features(embeddings)
        if batch_ids is not None and hasattr(self, "_batch_enc"):
            meta = np.hstack([meta, self._batch_enc.transform(batch_ids.reshape(-1, 1))])
        return self.meta_classifier_.predict_proba(meta)

    def get_method_importance(self) -> Dict[str, float]:
        """Normalized feature importance aggregated per method."""
        imp = self.meta_classifier_.feature_importances_
        C = self.n_cell_types
        scores = {name: imp[i*C:(i+1)*C].sum()
                  for i, name in enumerate(self.method_names)}
        total = sum(scores.values())
        return {k: v / total for k, v in sorted(scores.items(), key=lambda x: -x[1])} if total else scores
