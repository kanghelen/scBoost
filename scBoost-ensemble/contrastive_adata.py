"""
Technique 8: Cross-Batch Supervised Contrastive Learning — AnnData interface.
Reads concatenated embeddings from adata.obsm, labels/batch from adata.obs.
Stores refined embedding back into adata.obsm.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from anndata import AnnData
from typing import List, Optional


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class CrossBatchSupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, features, labels, batch_ids):
        B = features.shape[0]
        device = features.device
        sim = features @ features.T / self.tau

        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
        diff_batch = batch_ids.unsqueeze(0) != batch_ids.unsqueeze(1)
        not_self = ~torch.eye(B, dtype=torch.bool, device=device)
        positives = same_label & diff_batch & not_self

        has_pos = positives.any(dim=1)
        if not has_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        logits = sim - sim.max(dim=1, keepdim=True).values.detach()
        log_denom = torch.log((torch.exp(logits) * not_self.float()).sum(1, keepdim=True) + 1e-8)
        log_prob = logits - log_denom

        loss = -(positives.float() * log_prob).sum(1) / positives.float().sum(1).clamp(min=1)
        return loss[has_pos].mean()


class CrossBatchSupCon:
    def __init__(self, input_dim: int, hidden_dim: int = 128, proj_dim: int = 64,
                 temperature: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-4,
                 device: str = "auto"):
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
        self.proj = ProjectionHead(input_dim, hidden_dim, proj_dim).to(self.device)
        self.loss_fn = CrossBatchSupConLoss(temperature)
        self.lr, self.wd = lr, weight_decay

    def _batch_iter(self, b_enc, size):
        unique = np.unique(b_enc)
        per = max(32, size // len(unique))
        by_batch = {b: np.where(b_enc == b)[0] for b in unique}
        n_iter = max(1, len(b_enc) // size)
        for _ in range(n_iter):
            idx = np.concatenate([
                np.random.choice(v, min(per, len(v)), replace=False)
                for v in by_batch.values()
            ])[:size]
            yield idx

    def fit(self, adata: AnnData, obsm_keys: List[str],
            label_key: str = "cell_type", batch_key: str = "batch",
            epochs: int = 100, batch_size: int = 512, verbose: bool = True):
        """
        Train on concatenated embeddings from adata.obsm[obsm_keys].

        Parameters
        ----------
        adata : AnnData
        obsm_keys : list of str
            Keys in adata.obsm to concatenate as input features.
        label_key, batch_key : str
            Columns in adata.obs.
        """
        concat = np.hstack([np.asarray(adata.obsm[k]) for k in obsm_keys]).astype(np.float32)
        self.obsm_keys_ = obsm_keys

        le_y, le_b = LabelEncoder(), LabelEncoder()
        y_enc = le_y.fit_transform(np.asarray(adata.obs[label_key]))
        b_enc = le_b.fit_transform(np.asarray(adata.obs[batch_key]))

        # Rebuild projection if input dim changed
        if concat.shape[1] != self.proj.net[0].in_features:
            self.proj = ProjectionHead(concat.shape[1]).to(self.device)

        X = torch.FloatTensor(concat).to(self.device)
        Y = torch.LongTensor(y_enc).to(self.device)
        B = torch.LongTensor(b_enc).to(self.device)

        opt = torch.optim.AdamW(self.proj.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

        self.proj.train()
        for ep in range(epochs):
            losses = []
            for idx in self._batch_iter(b_enc, batch_size):
                t = torch.LongTensor(idx).to(self.device)
                loss = self.loss_fn(self.proj(X[t]), Y[t], B[t])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.proj.parameters(), 1.0)
                opt.step(); losses.append(loss.item())
            sched.step()
            if verbose and (ep + 1) % 20 == 0:
                print(f"  Epoch {ep+1}/{epochs} | Loss: {np.mean(losses):.4f}")
        return self

    @torch.no_grad()
    def transform(self, adata: AnnData, key_added: str = "X_supcon",
                  obsm_keys: Optional[List[str]] = None, chunk: int = 4096) -> np.ndarray:
        """
        Project adata and store result in adata.obsm[key_added].
        """
        keys = obsm_keys or self.obsm_keys_
        concat = np.hstack([np.asarray(adata.obsm[k]) for k in keys]).astype(np.float32)

        self.proj.eval()
        X = torch.FloatTensor(concat).to(self.device)
        parts = [self.proj(X[i:i+chunk]).cpu().numpy() for i in range(0, len(X), chunk)]
        emb = np.vstack(parts)
        adata.obsm[key_added] = emb
        return emb

    def fit_transform(self, adata: AnnData, obsm_keys: List[str],
                      label_key: str = "cell_type", batch_key: str = "batch",
                      key_added: str = "X_supcon", **kw) -> np.ndarray:
        self.fit(adata, obsm_keys, label_key, batch_key, **kw)
        return self.transform(adata, key_added, obsm_keys)
