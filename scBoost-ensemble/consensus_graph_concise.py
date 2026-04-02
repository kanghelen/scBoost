"""
Technique 10: Consensus Graph Embedding.
Builds k-NN graphs per method, combines via Jaccard/intersection/union,
optionally diffuses, then spectral-embeds the consensus graph.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Optional, Literal


class ConsensusGraphEmbedding:
    def __init__(
        self,
        n_neighbors: int = 30,
        n_components: int = 30,
        strategy: Literal["intersection", "weighted_union", "jaccard"] = "jaccard",
        jaccard_threshold: float = 0.4,
        method_weights: Optional[Dict[str, float]] = None,
        metric: str = "cosine",
        diffusion_time: int = 3,
        use_diffusion: bool = True,
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.strategy = strategy
        self.jaccard_threshold = jaccard_threshold
        self.method_weights = method_weights
        self.metric = metric
        self.diffusion_time = diffusion_time
        self.use_diffusion = use_diffusion

    def _knn_binary(self, X: np.ndarray) -> sparse.csr_matrix:
        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1, metric=self.metric, n_jobs=-1)
        nn.fit(X)
        _, idx = nn.kneighbors(X)
        idx = idx[:, 1:]
        n = X.shape[0]
        rows = np.repeat(np.arange(n), self.n_neighbors)
        A = sparse.csr_matrix((np.ones(len(rows)), (rows, idx.ravel())), shape=(n, n))
        return ((A + A.T) > 0).astype(float)

    def _knn_weighted(self, X: np.ndarray) -> sparse.csr_matrix:
        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1, metric=self.metric, n_jobs=-1)
        nn.fit(X)
        dist, idx = nn.kneighbors(X)
        dist, idx = dist[:, 1:], idx[:, 1:]
        sigma = max(np.median(dist[:, -1]), 1e-6)
        w = np.exp(-dist**2 / (2 * sigma**2))
        n = X.shape[0]
        rows = np.repeat(np.arange(n), self.n_neighbors)
        A = sparse.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))
        return (A + A.T) / 2

    def _consensus(self, graphs, method_names, embeddings_dict, batch_ids):
        if self.strategy == "intersection":
            C = graphs[0].copy()
            for g in graphs[1:]:
                C = C.multiply(g)
            return (C > 0).astype(float)

        elif self.strategy == "weighted_union":
            if self.method_weights:
                weights = [self.method_weights.get(n, 1.0) for n in method_names]
            elif batch_ids is not None:
                weights = [self._batch_entropy(embeddings_dict[n], batch_ids)
                           for n in method_names]
                s = sum(weights); weights = [w / s for w in weights]
            else:
                weights = [1.0 / len(graphs)] * len(graphs)
            C = sum(w * g for w, g in zip(weights, graphs))
            mx = C.max()
            return C / mx if mx > 0 else C

        else:  # jaccard
            support = sum(g for g in graphs)
            thresh = self.jaccard_threshold * len(graphs)
            mask = (support >= thresh).astype(float)
            return mask.multiply(support / len(graphs))

    def _diffuse(self, A: sparse.csr_matrix) -> sparse.csr_matrix:
        rs = np.array(A.sum(1)).ravel(); rs[rs == 0] = 1
        P = sparse.diags(1.0 / rs) @ A
        Pt = P.copy()
        for _ in range(self.diffusion_time - 1):
            Pt = Pt @ P
        return (Pt + Pt.T) / 2

    def _spectral(self, A: sparse.csr_matrix) -> np.ndarray:
        n = A.shape[0]
        deg = np.array(A.sum(1)).ravel(); deg[deg == 0] = 1e-10
        D_isq = sparse.diags(1.0 / np.sqrt(deg))
        A_norm = D_isq @ A @ D_isq
        k = min(self.n_components + 1, n - 1)
        vals, vecs = eigsh(A_norm, k=k, which="LM", maxiter=5000)
        order = np.argsort(-vals)
        vals, vecs = vals[order], vecs[:, order]
        emb = vecs[:, 1:self.n_components + 1]
        return emb * np.sqrt(np.abs(vals[1:self.n_components + 1]))

    def _batch_entropy(self, X, batch_ids, k=50):
        nn = NearestNeighbors(n_neighbors=k + 1, metric=self.metric, n_jobs=-1)
        nn.fit(X); _, idx = nn.kneighbors(X)
        ub = np.unique(batch_ids)
        ents = []
        for i in range(len(X)):
            _, c = np.unique(batch_ids[idx[i, 1:]], return_counts=True)
            p = c / c.sum()
            ents.append(-np.sum(p * np.log(p + 1e-10)))
        return np.mean(ents) / np.log(len(ub))

    def fit_transform(self, embeddings_dict: Dict[str, np.ndarray],
                      batch_ids: Optional[np.ndarray] = None) -> np.ndarray:
        names = list(embeddings_dict.keys())
        if self.strategy == "weighted_union":
            graphs = [self._knn_weighted(embeddings_dict[n]) for n in names]
        else:
            graphs = [self._knn_binary(embeddings_dict[n]) for n in names]

        C = self._consensus(graphs, names, embeddings_dict, batch_ids)
        if self.use_diffusion:
            C = self._diffuse(C)

        self.consensus_graph_ = C
        self.embedding_ = self._spectral(C)
        return self.embedding_

    def get_stats(self) -> dict:
        A = self.consensus_graph_
        deg = np.array(A.sum(1)).ravel()
        return dict(n_cells=A.shape[0], n_edges=A.nnz // 2,
                    mean_degree=deg.mean(), isolated=int((deg == 0).sum()))
