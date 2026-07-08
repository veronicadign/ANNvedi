import numpy as np
import sys
import os
import time
import random
import warnings

# Ensure the local directory is on python path for importing the compiled binary module
sys.path.insert(0, os.path.dirname(__file__))

import ivf_lsh_cpp

class Algorithm:
    def __init__(self):
        self._index = ivf_lsh_cpp.LSHIndex()
        self._best_query_params = {'n_probes': 0, 'n_probe_clusters': 8, 'refine_r': -1}

    def _evaluate_config(self, train_sub: np.ndarray, val_queries: np.ndarray, 
                         gt_neighbors: np.ndarray, k: int, config: tuple) -> tuple:
        """
        Evaluate a single candidate parameter configuration on the validation subset.
        Returns: (recall, query_time_ms, estimated_index_mem_mb)
        """
        n_tables, n_bits, n_clusters, n_probe_clusters, n_probes, refine_r = config
        try:
            # Fit transient index on subsample
            tmp_idx = ivf_lsh_cpp.LSHIndex()
            tmp_idx.fit(train_sub, n_tables, n_bits, n_clusters)

            # Query and measure average time per query
            t0 = time.perf_counter()
            results = []
            for q in val_queries:
                res = tmp_idx.query(q, k=k, n_probes=n_probes, n_probe_clusters=n_probe_clusters, refine_r=refine_r)
                results.append(res[:k])
            q_time = (time.perf_counter() - t0) * 1000 / len(val_queries)

            # Compute recall@k
            recalls = [len(set(gt_neighbors[i]).intersection(results[i])) / float(k) for i in range(len(val_queries))]
            recall = np.mean(recalls)

            # Estimate index memory in MB
            idx_mem = (n_tables * n_clusters * (2 ** n_bits) * 4) / (1024.0 * 1024.0)

            return recall, q_time, idx_mem
        except Exception:
            return 0.0, 1000.0, 1000.0

    def fit(self, train: np.ndarray, **index_params) -> None:
        train = np.asarray(train, dtype=np.float32)
        npts, dim = train.shape

        scenario = os.environ.get("SCENARIO_NAME", "high_recall").lower()
        k = int(os.environ.get("QUERY_K", 10))
        print(f"[Algorithm] Target scenario detected: {scenario} | Query K: {k}")

        # If data is small, bypass GP tuning
        if npts < 10000:
            print("[Algorithm] Small dataset, bypassing GP tuning...")
            n_tables = int(index_params.get('n_tables', 10))
            n_bits = int(index_params.get('n_bits', 8))
            n_clusters = int(index_params.get('n_clusters', 64))
            self._index.fit(train, n_tables, n_bits, n_clusters)
            self._best_query_params = {
                'n_probes': int(index_params.get('n_probes', 4)),
                'n_probe_clusters': int(index_params.get('n_probe_clusters', 8)),
                'refine_r': int(index_params.get('refine_r', -1))
            }
            return

        # Perform GP parameter tuning
        print("[Algorithm] Running GP auto-tuning on training subset...")
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel
            from sklearn.preprocessing import StandardScaler
            warnings.filterwarnings('ignore')

            # Create train/val split safely
            indices = np.arange(npts)
            np.random.seed(42)
            np.random.shuffle(indices)

            n_val = 50
            n_train_sub = min(15000, npts - n_val)
            
            val_idx = indices[:n_val]
            train_sub_idx = indices[n_val:n_val + n_train_sub]
            
            train_sub = train[train_sub_idx]
            val_queries = train[val_idx]

            # Compute exact ground truth for validation queries
            print("[Algorithm] Computing ground truth for validation queries...")
            gt_neighbors = []
            for q in val_queries:
                dists = np.linalg.norm(train_sub - q, axis=1)
                gt_neighbors.append(np.argsort(dists)[:k])
            gt_neighbors = np.array(gt_neighbors)

            # Generate configuration samples, starting with known anchors for high/mid/low recall coverage
            configs = [
                (30, 8, 256, 16, 8, int(3 * k)),   # Anchor 1: High recall
                (20, 10, 512, 32, 16, int(4 * k)), # Anchor 2: Very high recall
                (5, 6, 128, 8, 4, int(k)),         # Anchor 3: Low memory / fast
                (10, 8, 256, 12, 4, int(1.5 * k))  # Anchor 4: Balanced
            ]

            random.seed(42)
            while len(configs) < 25:
                n_tables = int(random.choice([5, 10, 15, 20, 30]))
                n_bits = int(random.choice([6, 8, 10, 12]))
                n_clusters = int(random.choice([128, 256, 512]))
                n_probe_clusters = int(random.choice([4, 8, 16, 24, 32]))
                n_probes = int(random.choice([0, 4, 8, 12, 16, 20]))
                refine_r = int(random.choice([int(k), int(1.5 * k), int(2.0 * k), int(3.0 * k)]))

                if n_probe_clusters > n_clusters:
                    n_probe_clusters = n_clusters
                cfg = (n_tables, n_bits, n_clusters, n_probe_clusters, n_probes, refine_r)
                if cfg not in configs:
                    configs.append(cfg)

            X_data = []
            y_recall = []
            y_time = []
            y_mem = []

            for config in configs:
                recall, q_time, idx_mem = self._evaluate_config(train_sub, val_queries, gt_neighbors, k, config)
                X_data.append(config)
                y_recall.append(recall)
                y_time.append(q_time)
                y_mem.append(idx_mem)

            X = np.array(X_data)
            y_rec = np.array(y_recall)
            y_t = np.log(np.array(y_time) + 1e-5)
            y_m = np.log(np.array(y_mem) + 1e-5)

            # Fit GPs
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            gp_recall = GaussianProcessRegressor(
                kernel=Matern(length_scale=[1.5]*6, nu=1.5) + WhiteKernel(noise_level=1e-4),
                random_state=42
            ).fit(X_scaled, y_rec)

            gp_time = GaussianProcessRegressor(
                kernel=Matern(length_scale=[1.5]*6, nu=1.5) + WhiteKernel(noise_level=1e-3),
                random_state=42
            ).fit(X_scaled, y_t)

            gp_mem = GaussianProcessRegressor(
                kernel=Matern(length_scale=[1.5]*6, nu=1.5) + WhiteKernel(noise_level=1e-3),
                random_state=42
            ).fit(X_scaled, y_m)

            # Grid search candidates
            candidate_tables = [5, 10, 15, 20, 25, 30]
            candidate_bits = [6, 8, 10, 12]
            candidate_clusters = [128, 256, 512]
            candidate_probe_cls = [4, 8, 16, 24, 32]
            candidate_probes = [0, 4, 8, 12, 16, 20]
            candidate_refines = [int(k), int(1.5 * k), int(2.0 * k), int(3.0 * k), int(4.0 * k)]

            search_space = []
            for t in candidate_tables:
                for b in candidate_bits:
                    for c in candidate_clusters:
                        for pc in candidate_probe_cls:
                            if pc <= c:
                                for p in candidate_probes:
                                    for r in candidate_refines:
                                        search_space.append([t, b, c, pc, p, r])

            search_space = np.array(search_space)
            search_space_scaled = scaler.transform(search_space)

            pred_recs = np.clip(gp_recall.predict(search_space_scaled), 0.0, 1.0)
            pred_times = np.exp(gp_time.predict(search_space_scaled))
            pred_mems = np.exp(gp_mem.predict(search_space_scaled))

            best_cfg = None

            # Fallback to empirical best if GP predictions are flat (std < 0.02)
            if np.std(pred_recs) < 0.02:
                print("[Algorithm] GP predictions are flat. Falling back to empirical best.")
                if scenario == 'fast':
                    idx_fast = np.where(y_rec >= 0.80)[0]
                    if len(idx_fast) > 0:
                        best_idx = idx_fast[np.argmin(y_time[idx_fast])]
                        best_cfg = X[best_idx]
                elif scenario == 'memory':
                    idx_mem = np.where(y_rec >= 0.95)[0]
                    if len(idx_mem) > 0:
                        best_idx = idx_mem[np.argmin(y_mem[idx_mem])]
                        best_cfg = X[best_idx]
                
                if best_cfg is None:
                    idx_hr = np.where(y_rec >= 0.95)[0]
                    if len(idx_hr) > 0:
                        best_idx = idx_hr[np.argmin(y_time[idx_hr])]
                        best_cfg = X[best_idx]
                    else:
                        best_idx = np.argmax(y_rec)
                        best_cfg = X[best_idx]
            else:
                # Use GP predictions to search grid
                if scenario == 'fast':
                    # Target: Recall >= 80%, minimize query time
                    idx_fast = np.where(pred_recs >= 0.80)[0]
                    if len(idx_fast) > 0:
                        best_fast_idx = idx_fast[np.argmin(pred_times[idx_fast])]
                        best_cfg = search_space[best_fast_idx]
                elif scenario == 'memory':
                    # Target: Recall >= 95%, minimize memory
                    idx_mem = np.where(pred_recs >= 0.95)[0]
                    if len(idx_mem) > 0:
                        best_mem_idx = idx_mem[np.argmin(pred_mems[idx_mem])]
                        best_cfg = search_space[best_mem_idx]
                
                # Default fallback for high_recall or empty subset matches
                if best_cfg is None:
                    idx_hr = np.where(pred_recs >= 0.95)[0]
                    if len(idx_hr) > 0:
                        best_hr_idx = idx_hr[np.argmin(pred_times[idx_hr])]
                        best_cfg = search_space[best_hr_idx]
                    else:
                        best_cfg = search_space[np.argmax(pred_recs)]

            opt_tables, opt_bits, opt_clusters, opt_probe_cl, opt_probes, opt_refine_r = map(int, best_cfg)
            print(f"[Algorithm] GP Auto-tuning selected config: tables={opt_tables}, bits={opt_bits}, clusters={opt_clusters}, probe_cl={opt_probe_cl}, probes={opt_probes}, refine_r={opt_refine_r}")

        except Exception as e:
            print(f"[Algorithm] GP tuning failed: {e}. Falling back to default parameters.")
            opt_tables = int(index_params.get('n_tables', 15))
            opt_bits = int(index_params.get('n_bits', 8))
            opt_clusters = int(index_params.get('n_clusters', 256))
            opt_probe_cl = int(index_params.get('n_probe_clusters', 8))
            opt_probes = int(index_params.get('n_probes', 4))
            opt_refine_r = int(index_params.get('refine_r', -1))

        # Fit the final index on full dataset
        print(f"[Algorithm] Fitting final index on full dataset (size={npts}) ...")
        self._index.fit(train, opt_tables, opt_bits, opt_clusters)
        self._best_query_params = {
            'n_probes': opt_probes,
            'n_probe_clusters': opt_probe_cl,
            'refine_r': opt_refine_r
        }

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        n_probes = self._best_query_params['n_probes']
        n_probe_clusters = self._best_query_params['n_probe_clusters']
        refine_r = self._best_query_params['refine_r']
        return self._index.query(query, k, n_probes, n_probe_clusters, refine_r)

    def get_n_distances(self) -> int:
        return self._index.total_distances_count()
