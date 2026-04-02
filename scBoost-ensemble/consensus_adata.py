"""
Technique 10: Consensus Graph Embedding — AnnData interface.
Reads per-method embeddings from adata.obsm, stores consensus embedding back.
Supports GPU acceleration via FAISS + CuPy when device="cuda".
"""

import time
import numpy as np
from scipy import sparse as sp_sparse
from scipy.sparse.linalg import eigsh as sp_eigsh
from anndata import AnnData
from typing import List, Dict, Optional, Literal


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
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        device : str
            "cpu" or "cuda". GPU path uses FAISS for k-NN and CuPy for
            sparse ops / spectral embedding. Falls back to CPU if unavailable.
        """
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.strategy = strategy
        self.jaccard_threshold = jaccard_threshold
        self.method_weights = method_weights
        self.metric = metric
        self.diffusion_time = diffusion_time
        self.use_diffusion = use_diffusion
        self._use_gpu = False

        if device == "cuda":
            try:
                import faiss
                import cupy as cp
                import cupyx.scipy.sparse as cp_sparse
                from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh
                self._faiss = faiss
                self._cp = cp
                self._cp_sparse = cp_sparse
                self._cp_eigsh = cp_eigsh
                self._gpu_res = faiss.StandardGpuResources()
                self._use_gpu = True
                print("ConsensusGraphEmbedding: using GPU (FAISS + CuPy)")
            except ImportError as e:
                print(f"ConsensusGraphEmbedding: GPU unavailable ({e}), falling back to CPU")

    # --- k-NN ---

    def _knn_faiss(self, X, k):
        X = np.ascontiguousarray(X, dtype=np.float32)
        d = X.shape[1]
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1
            X = X / norms
            index = self._faiss.IndexFlatIP(d)
        else:
            index = self._faiss.IndexFlatL2(d)
        gpu_index = self._faiss.index_cpu_to_gpu(self._gpu_res, 0, index)
        gpu_index.add(X)
        dist, idx = gpu_index.search(X, k + 1)
        if self.metric == "cosine":
            dist = 1.0 - dist
        return dist[:, 1:], idx[:, 1:]

    def _knn_sklearn(self, X, k):
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric=self.metric, n_jobs=-1)
        nn.fit(X)
        dist, idx = nn.kneighbors(X)
        return dist[:, 1:], idx[:, 1:]

    def _knn(self, X, k=None):
        k = k or self.n_neighbors
        if self._use_gpu:
            return self._knn_faiss(X, k)
        return self._knn_sklearn(X, k)

    def _knn_binary(self, X):
        _, idx = self._knn(X)
        n = X.shape[0]
        rows = np.repeat(np.arange(n), self.n_neighbors)
        A = sp_sparse.csr_matrix((np.ones(len(rows)), (rows, idx.ravel())), shape=(n, n))
        return ((A + A.T) > 0).astype(float)

    def _knn_weighted(self, X):
        dist, idx = self._knn(X)
        sigma = max(np.median(dist[:, -1]), 1e-6)
        w = np.exp(-dist**2 / (2 * sigma**2))
        n = X.shape[0]
        rows = np.repeat(np.arange(n), self.n_neighbors)
        A = sp_sparse.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))
        return (A + A.T) / 2

    def _batch_entropy(self, X, batch_ids, k=50):
        _, idx = self._knn(X, k=k)
        ub = np.unique(batch_ids)
        ents = []
        for i in range(len(X)):
            _, c = np.unique(batch_ids[idx[i]], return_counts=True)
            p = c / c.sum()
            ents.append(-np.sum(p * np.log(p + 1e-10)))
        return np.mean(ents) / np.log(len(ub))

    # --- Consensus ---

    def _consensus(self, graphs, keys, embs_dict, batch_ids):
        K = len(graphs)
        if self.strategy == "intersection":
            C = graphs[0].copy()
            for g in graphs[1:]:
                C = C.multiply(g)
            return (C > 0).astype(float)

        elif self.strategy == "weighted_union":
            if self.method_weights:
                wts = [self.method_weights.get(k, 1.0) for k in keys]
            elif batch_ids is not None:
                wts = [self._batch_entropy(embs_dict[k], batch_ids) for k in keys]
                s = sum(wts); wts = [w / s for w in wts]
            else:
                wts = [1.0 / K] * K
            C = sum(w * g for w, g in zip(wts, graphs))
            mx = C.max()
            return C / mx if mx > 0 else C

        else:  # jaccard
            support = sum(graphs)
            mask = (support >= self.jaccard_threshold * K).astype(float)
            return mask.multiply(support / K)

    # --- Diffusion & Spectral ---

    def _to_gpu_sparse(self, A):
        cp = self._cp
        return self._cp_sparse.csr_matrix(
            (cp.array(A.data), cp.array(A.indices), cp.array(A.indptr)), shape=A.shape)

    def _diffuse(self, A):
        if self._use_gpu:
            cp = self._cp
            A_g = self._to_gpu_sparse(A)
            rs = cp.array(A_g.sum(1)).ravel(); rs[rs == 0] = 1
            P = self._cp_sparse.diags(1.0 / rs) @ A_g
            Pt = P.copy()
            for _ in range(self.diffusion_time - 1):
                Pt = Pt @ P
            result = (Pt + Pt.T) / 2
            result = sp_sparse.csr_matrix(
                (result.data.get(), result.indices.get(), result.indptr.get()), shape=result.shape)
            return result
        else:
            rs = np.array(A.sum(1)).ravel(); rs[rs == 0] = 1
            P = sp_sparse.diags(1.0 / rs) @ A
            Pt = P.copy()
            for _ in range(self.diffusion_time - 1):
                Pt = Pt @ P
            return (Pt + Pt.T) / 2

    def _spectral(self, A):
        n = A.shape[0]
        k = min(self.n_components + 1, n - 1)

        if self._use_gpu:
            cp = self._cp
            A_g = self._to_gpu_sparse(A)
            deg = cp.array(A_g.sum(1)).ravel(); deg[deg == 0] = 1e-10
            D_isq = self._cp_sparse.diags(1.0 / cp.sqrt(deg))
            A_norm = D_isq @ A_g @ D_isq
            vals, vecs = self._cp_eigsh(A_norm, k=k, which="LM", maxiter=5000)
            vals, vecs = vals.get(), vecs.get()
        else:
            deg = np.array(A.sum(1)).ravel(); deg[deg == 0] = 1e-10
            D_isq = sp_sparse.diags(1.0 / np.sqrt(deg))
            A_norm = D_isq @ A @ D_isq
            vals, vecs = sp_eigsh(A_norm, k=k, which="LM", maxiter=5000)

        order = np.argsort(-vals)
        vals, vecs = vals[order], vecs[:, order]
        emb = vecs[:, 1:self.n_components + 1]
        return emb * np.sqrt(np.abs(vals[1:self.n_components + 1]))

    # --- Main entry ---

    def fit_transform(
        self,
        adata: AnnData,
        obsm_keys: List[str],
        batch_key: str = "batch",
        key_added: str = "X_consensus",
    ) -> np.ndarray:
        """
        Build consensus graph from adata.obsm[obsm_keys] and store
        the spectral embedding in adata.obsm[key_added].
        """
        t_start = time.time()
        backend = "GPU (FAISS+CuPy)" if self._use_gpu else "CPU (sklearn+scipy)"
        print(f"ConsensusGraph: {adata.n_obs} cells, {len(obsm_keys)} methods, "
              f"strategy={self.strategy}, device={backend}")

        embs = {k: np.asarray(adata.obsm[k]) for k in obsm_keys}
        batch_ids = np.asarray(adata.obs[batch_key]) if batch_key in adata.obs else None

        t0 = time.time()
        for ki, k in enumerate(obsm_keys):
            print(f"  Building k-NN graph for {k} ({ki+1}/{len(obsm_keys)})...")
        if self.strategy == "weighted_union":
            graphs = [self._knn_weighted(embs[k]) for k in obsm_keys]
        else:
            graphs = [self._knn_binary(embs[k]) for k in obsm_keys]
        print(f"  k-NN graphs done ({time.time()-t0:.1f}s)")

        t0 = time.time()
        C = self._consensus(graphs, obsm_keys, embs, batch_ids)
        print(f"  Consensus done ({time.time()-t0:.1f}s)")

        if self.use_diffusion:
            t0 = time.time()
            C = self._diffuse(C)
            print(f"  Diffusion done ({time.time()-t0:.1f}s)")

        t0 = time.time()
        self.consensus_graph_ = C
        emb = self._spectral(C)
        print(f"  Spectral embedding done ({time.time()-t0:.1f}s)")

        adata.obsm[key_added] = emb
        self.embedding_ = emb
        print(f"ConsensusGraph complete ({time.time()-t_start:.1f}s total)")
        return emb

    def get_stats(self) -> dict:
        A = self.consensus_graph_
        deg = np.array(A.sum(1)).ravel()
        return dict(n_cells=A.shape[0], n_edges=A.nnz // 2,
                    mean_degree=deg.mean(), isolated=int((deg == 0).sum()))
