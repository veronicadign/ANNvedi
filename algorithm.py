import numpy as np

# Backend modules are imported lazily inside fit(): a backend that is not
# installed/built only fails if a scenario actually selects it, instead of
# breaking every import of this module.
#   linear     -> linear_ann.wrapper.LinearANN   (linear_ann_cpp,   built by setup.py)
#   lsh        -> lsh_ann.wrapper.LSHANN         (lsh_cpp_optimized, built by setup.py)
#   hnsw       -> hnsw_cpp.HNSWIndex             (built by setup.py)
#   multiprobe -> multiprobe_lsh_cpp.MultiProbeLSH (built by setup.py)


class Algorithm:

    def __init__(self):
        self._backend = 'linear'
        self._index = None

    def fit(self, train: np.ndarray, **index_params) -> None:
        ds_size = index_params.pop('ds_size', None)
        if ds_size is not None:
            train = train[:int(ds_size)]
        
        self._backend = index_params.pop('backend', 'linear').lower()
        
        if self._backend == 'lsh':
            from lsh_ann.wrapper import LSHANN
            self._index = LSHANN(**index_params)
            self._index.fit(train)
        elif self._backend in ('ivf_lsh', 'ivf'):
            # per-dimension SQ8 IVF-LSH fork (module built from competitors/ivf_lsh/src)
            import ivf_lsh_cpp
            self._index = ivf_lsh_cpp.LSHIndex()
            n_tables = int(index_params.get('n_tables', 10))
            n_bits = int(index_params.get('n_bits', 8))
            n_clusters = int(index_params.get('n_clusters', 256))
            train_f32 = np.asarray(train, dtype=np.float32)
            self._index.fit(train_f32, n_tables, n_bits, n_clusters)
        elif self._backend == 'hnsw':
            import hnsw_cpp
            self._index = hnsw_cpp.HNSWIndex()
            M = int(index_params.get('M', 16))
            ef_construction = int(index_params.get('ef_construction', 100))
            mode = str(index_params.get('mode', 'float'))
            # Default OFF: with the competition's k=100, hnsw floors ef at k, so
            # recall already saturates and diversity edges only cost QPS/build
            # time (measured 2026-08-10: qps -15..-40%, build 2-4x). The offline
            # tuner sweeps it per dataset; enable via index_params if it wins.
            heuristic = bool(index_params.get('heuristic', False))
            train_f32 = np.asarray(train, dtype=np.float32)
            self._index.fit(train_f32, M, ef_construction, mode, heuristic)
        elif self._backend == 'multiprobe':
            import multiprobe_lsh_cpp
            self._index = multiprobe_lsh_cpp.MultiProbeLSH()
            n_tables = int(index_params.get('n_tables', 10))
            n_bits = int(index_params.get('n_bits', 8))
            train_f32 = np.asarray(train, dtype=np.float32)
            self._index.fit(train_f32, n_tables, n_bits)
        else:
            from linear_ann.wrapper import LinearANN
            self._index = LinearANN()
            self._index.fit(train)

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        if self._backend == 'lsh':
            refine_r = int(query_params.pop('refine_r', -1))
            return self._index.query(query, k, refine_r=refine_r, **query_params)
        elif self._backend in ('ivf_lsh', 'ivf'):
            n_probes = int(query_params.get('n_probes', 0))
            n_probe_clusters = int(query_params.get('n_probe_clusters', 8))
            refine_r = int(query_params.get('refine_r', -1))
            query_f32 = np.asarray(query, dtype=np.float32)
            return self._index.query(query_f32, k, n_probes, n_probe_clusters, refine_r)
        elif self._backend == 'hnsw':
            ef = int(query_params.get('ef', 50))
            refine_r = int(query_params.get('refine_r', -1))
            query_f32 = np.asarray(query, dtype=np.float32)
            return self._index.query(query_f32, k, ef, refine_r)
        elif self._backend == 'multiprobe':
            probe_depth = int(query_params.get('probe_depth', 1))
            query_f32 = np.asarray(query, dtype=np.float32)
            return self._index.query(query_f32, k, probe_depth)
        else:
            return self._index.query(query, k)

    def get_n_distances(self) -> int:
        if self._index is None:
            return 0
        if hasattr(self._index, 'total_distances_count'):
            return self._index.total_distances_count()
        return 0
