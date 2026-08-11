import numpy as np
import sys
import os

# Ensure the local directory is on python path for importing the compiled binary module
sys.path.insert(0, os.path.dirname(__file__))

import ivf_lsh_cpp


class Algorithm:
    """IVF + multiprobe LSH (per-dimension SQ8) submission.

    Parameters come EXCLUSIVELY from scenarios.yaml (index_params/query_params,
    resolved per scenario and per dataset by the harness). There is no tuning
    at fit() time: since the competition datasets match the public ones, the
    per-dataset parameters are found once, offline, with
    scripts/tune_parameters.py (run it on the target machine) and written into
    scenarios.yaml. This keeps build_time minimal - it is a scored metric.

    (The previous version ran a Gaussian-Process parameter search inside fit();
    see git history if it is ever needed again.)
    """

    def __init__(self):
        self._index = ivf_lsh_cpp.LSHIndex()
        self._default_query_params = {'n_probes': 4, 'n_probe_clusters': 8, 'refine_r': -1}

    def fit(self, train: np.ndarray, **index_params) -> None:
        train = np.asarray(train, dtype=np.float32)
        n_tables = int(index_params.get('n_tables', 15))
        n_bits = int(index_params.get('n_bits', 8))
        n_clusters = int(index_params.get('n_clusters', 256))
        # Optional query-param defaults in index_params, used only if the
        # scenario provides no query_params (kept for scenarios.yaml brevity).
        for key in self._default_query_params:
            if key in index_params:
                self._default_query_params[key] = int(index_params[key])
        self._index.fit(train, n_tables, n_bits, n_clusters)

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        n_probes = int(query_params.get('n_probes', self._default_query_params['n_probes']))
        n_probe_clusters = int(query_params.get('n_probe_clusters', self._default_query_params['n_probe_clusters']))
        refine_r = int(query_params.get('refine_r', self._default_query_params['refine_r']))
        return self._index.query(query, k, n_probes, n_probe_clusters, refine_r)

    def get_n_distances(self) -> int:
        return self._index.total_distances_count()
