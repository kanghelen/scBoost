"""
Technique 8: Cross-Batch Supervised Contrastive Learning.
Learns a refined embedding where same-type cells from different batches
are pulled together and different-type cells are pushed apart.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from typing import Optional


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
    """SupCon loss where positives must share label but differ in batch."""

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

        logits_max = sim.max(dim=1, keepdim=True).values.detach()
        logits = sim - logits_max
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
        self.lr = lr
        self.wd = weight_decay

    def _make_batches(self, batch_ids_enc: np.ndarray, size: int):
        """Yield mini-batch indices ensuring all batches represented."""
        unique = np.unique(batch_ids_enc)
        per = max(32, size // len(unique))
        idx_by_batch = {b: np.where(batch_ids_enc == b)[0] for b in unique}
        n_iter = max(1, len(batch_ids_enc) // size)
        for _ in range(n_iter):
            idx = np.concatenate([
                np.random.choice(v, min(per, len(v)), replace=False)
                for v in idx_by_batch.values()
            ])[:size]
            yield idx

    def fit(self, embeddings: np.ndarray, labels: np.ndarray,
            batch_ids: np.ndarray, epochs: int = 100, batch_size: int = 512,
            verbose: bool = True):
        le_y, le_b = LabelEncoder(), LabelEncoder()
        y_enc = le_y.fit_transform(labels)
        b_enc = le_b.fit_transform(batch_ids)

        X = torch.FloatTensor(embeddings).to(self.device)
        Y = torch.LongTensor(y_enc).to(self.device)
        B = torch.LongTensor(b_enc).to(self.device)

        opt = torch.optim.AdamW(self.proj.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

        self.proj.train()
        for ep in range(epochs):
            losses = []
            for idx in self._make_batches(b_enc, batch_size):
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
    def transform(self, embeddings: np.ndarray, chunk: int = 4096) -> np.ndarray:
        self.proj.eval()
        X = torch.FloatTensor(embeddings).to(self.device)
        parts = [self.proj(X[i:i+chunk]).cpu().numpy() for i in range(0, len(X), chunk)]
        return np.vstack(parts)

    def fit_transform(self, embeddings, labels, batch_ids, **kw):
        self.fit(embeddings, labels, batch_ids, **kw)
        return self.transform(embeddings)
