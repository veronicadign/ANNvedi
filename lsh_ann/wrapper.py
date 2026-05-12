import numpy as np


class LSHANN:
    def __init__(self, n_tables=10, n_bits=8):
        self._n_tables = int(n_tables)
        self._n_bits = int(n_bits)
        self._projections = []   # list of (n_bits, D) arrays, one per table
        self._tables = []        # list of dict {hash_key: [indices]}
        self._data = None
        self._n_distances = 0

    def fit(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("data must be 2D")
        self._data = data
        n, d = data.shape
        rng = np.random.default_rng(42)

        self._projections = []
        self._tables = []
        for _ in range(self._n_tables):
            # random unit projection vectors: (n_bits, D)
            proj = rng.standard_normal((self._n_bits, d)).astype(np.float32)
            norms = np.linalg.norm(proj, axis=1, keepdims=True)
            proj /= np.where(norms > 0, norms, 1.0)
            self._projections.append(proj)

            # hash all points: (N, n_bits) → binary → tuple key
            codes = (data @ proj.T) >= 0   # (N, n_bits) bool
            table: dict = {}
            for i, code in enumerate(codes):
                key = code.tobytes()
                if key not in table:
                    table[key] = []
                table[key].append(i)
            self._tables.append(table)

    def _hash_query(self, q: np.ndarray, proj: np.ndarray) -> bytes:
        return ((q @ proj.T) >= 0).tobytes()

    def query(self, q: np.ndarray, k: int) -> np.ndarray:
        if self._data is None:
            raise RuntimeError("index not trained")
        q = np.asarray(q, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != self._data.shape[1]:
            raise ValueError("query shape mismatch")

        candidates: set = set()
        for proj, table in zip(self._projections, self._tables):
            key = self._hash_query(q, proj)
            if key in table:
                candidates.update(table[key])

        # fall back to all points if not enough candidates
        if len(candidates) < k:
            candidates = set(range(len(self._data)))

        cand_list = list(candidates)
        cand_data = self._data[cand_list]
        diff = cand_data - q
        dists = (diff * diff).sum(axis=1)
        self._n_distances += len(cand_list)

        if k >= len(cand_list):
            top_idx = np.argsort(dists)[:k]
        else:
            top_idx = np.argpartition(dists, k - 1)[:k]

        result = np.array([cand_list[i] for i in top_idx], dtype=np.int64)
        return result

    def total_distances_count(self) -> int:
        return self._n_distances
