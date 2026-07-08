#!/usr/bin/env python3
import sys
import os
import time
import h5py
import numpy as np
import random
import yaml
import warnings
import resource
import argparse

# Suppress convergence warnings from GP fitting
warnings.filterwarnings('ignore')

def get_memory_usage_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def main():
    parser = argparse.ArgumentParser(description="GP parameter tuner for IVF-LSH on a single dataset")
    parser.add_argument("--dataset", type=str, default="../dataset/yahoo-minilm-public.hdf5",
                        help="Path to HDF5 dataset file")
    parser.add_argument("--num_configs", type=int, default=30,
                        help="Number of random configurations to evaluate for GP training")
    parser.add_argument("--train_size", type=int, default=50000,
                        help="Subsample size for training data")
    parser.add_argument("--test_size", type=int, default=100,
                        help="Subsample size for query evaluation")
    args = parser.parse_args()

    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        # Try local path fallback
        fallback = os.path.join("datasets", os.path.basename(dataset_path))
        if os.path.exists(fallback):
            dataset_path = fallback
        else:
            print(f"❌ Dataset not found at {dataset_path}")
            sys.exit(1)

    print(f"==================================================")
    print(f"🎯 IVF-LSH GP Parameter Tuner")
    print(f"📂 Dataset: {dataset_path}")
    print(f"==================================================")

    # Add necessary paths to sys.path
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(root_dir, 'competitors/ivf_lsh'))
    sys.path.insert(0, root_dir)

    try:
        from competitors.ivf_lsh.algorithm import Algorithm
    except ImportError as e:
        print("❌ Error: Could not import ivf_lsh Algorithm wrapper.")
        print(f"Make sure ivf_lsh_cpp is compiled using 'pip install .' in competitors/ivf_lsh.")
        print(f"Detail: {e}")
        sys.exit(1)

    print("Loading dataset...")
    with h5py.File(dataset_path, 'r') as f:
        train = f['train'][:]
        test = f['test'][:]
        if 'neighbors' in f:
            gt_neighbors = f['neighbors'][:]
        else:
            gt_neighbors = None

    n_train = min(args.train_size, len(train))
    n_test = min(args.test_size, len(test))
    train_sub = train[:n_train]
    test_sub = test[:n_test]

    print(f"Subsampled train size: {n_train} | test size: {n_test}")

    # Compute ground truth locally for recall if not available in dataset
    if gt_neighbors is None or len(gt_neighbors) < n_test:
        print("Computing exact ground truth (Linear Scan)...")
        from linear_ann.wrapper import LinearANN
        linear_index = LinearANN()
        linear_index.fit(train_sub)
        gt_list = []
        for q in test_sub:
            gt_list.append(linear_index.query(q, 10))  # standard k=10
        gt_neighbors = np.array(gt_list)
    else:
        gt_neighbors = gt_neighbors[:n_test, :10]

    def measure_config(n_tables, n_bits, n_clusters, n_probe_clusters, n_probes, refine_r):
        try:
            # Force garbage collection to clean up memory
            import gc
            gc.collect()

            mem_before = get_memory_usage_mb()
            
            lsh = Algorithm()
            lsh.fit(train_sub, n_tables=n_tables, n_bits=n_bits, n_clusters=n_clusters)
            
            mem_after = get_memory_usage_mb()
            index_mem = max(0.1, mem_after - mem_before)

            t0 = time.perf_counter()
            results = []
            for q in test_sub:
                res = lsh.query(q, k=10, n_probes=n_probes, n_probe_clusters=n_probe_clusters, refine_r=refine_r)
                results.append(res[:10])
            query_time = (time.perf_counter() - t0) * 1000 / n_test # ms per query

            # Compute recall@10
            recalls = []
            for i in range(n_test):
                true_set = set(gt_neighbors[i])
                pred_set = set(results[i])
                intersection = true_set.intersection(pred_set)
                recalls.append(len(intersection) / 10.0)
            avg_recall = np.mean(recalls)

            return avg_recall, query_time, index_mem
        except Exception as e:
            return 0.0, 1000.0, 1000.0

    # Search space definition
    np.random.seed(42)
    random.seed(42)

    configs = []
    print(f"\nGenerating {args.num_configs} random parameter configurations...")
    while len(configs) < args.num_configs:
        n_tables = int(random.choice([5, 10, 15, 20, 25, 30]))
        n_bits = int(random.choice([6, 8, 10, 12]))
        n_clusters = int(random.choice([128, 256, 512]))
        n_probe_clusters = int(random.choice([4, 8, 16, 24, 32]))
        n_probes = int(random.choice([0, 4, 8, 12, 16, 20]))
        refine_r = int(random.choice([20, 50, 100, 150, 200]))

        if n_probe_clusters > n_clusters:
            n_probe_clusters = n_clusters

        cfg = (n_tables, n_bits, n_clusters, n_probe_clusters, n_probes, refine_r)
        if cfg not in configs:
            configs.append(cfg)

    X_data = []
    y_recall = []
    y_time = []
    y_mem = []

    print("\nEvaluating configurations...")
    for idx, (n_tables, n_bits, n_clusters, n_probe_clusters, n_probes, refine_r) in enumerate(configs):
        print(f"Config {idx+1}/{len(configs)}: tables={n_tables}, bits={n_bits}, clusters={n_clusters}, probe_cl={n_probe_clusters}, probes={n_probes}, refine_r={refine_r} ... ", end="", flush=True)
        recall, q_time, idx_mem = measure_config(n_tables, n_bits, n_clusters, n_probe_clusters, n_probes, refine_r)
        print(f"Recall={recall:.4f}, Time={q_time:.2f} ms, Mem={idx_mem:.1f} MB")

        X_data.append([n_tables, n_bits, n_clusters, n_probe_clusters, n_probes, refine_r])
        y_recall.append(recall)
        y_time.append(q_time)
        y_mem.append(idx_mem)

    X = np.array(X_data)
    y_rec = np.array(y_recall)
    y_t = np.log(np.array(y_time))
    y_m = np.log(np.array(y_mem))

    # Fit Gaussian Processes
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nFitting Gaussian Process Regressors...")
    gp_recall = GaussianProcessRegressor(
        kernel=Matern(length_scale=[1.0]*6, nu=2.5) + WhiteKernel(noise_level=1e-4),
        n_restarts_optimizer=10, random_state=42
    )
    gp_recall.fit(X_scaled, y_rec)

    gp_time = GaussianProcessRegressor(
        kernel=Matern(length_scale=[1.0]*6, nu=2.5) + WhiteKernel(noise_level=1e-3),
        n_restarts_optimizer=10, random_state=42
    )
    gp_time.fit(X_scaled, y_t)

    gp_mem = GaussianProcessRegressor(
        kernel=Matern(length_scale=[1.0]*6, nu=2.5) + WhiteKernel(noise_level=1e-3),
        n_restarts_optimizer=10, random_state=42
    )
    gp_mem.fit(X_scaled, y_m)

    print("Gaussian Processes trained successfully.")

    # Dense grid search for optimal configurations
    print("\nSearching parameter grid for optimal configurations...")
    candidate_tables = [5, 10, 15, 20, 25, 30]
    candidate_bits = [6, 8, 10, 12]
    candidate_clusters = [128, 256, 512]
    candidate_probe_cls = [4, 8, 16, 24, 32]
    candidate_probes = [0, 4, 8, 12, 16, 20]
    candidate_refines = [20, 50, 100, 150, 200]

    all_search_space = []
    for t in candidate_tables:
        for b in candidate_bits:
            for c in candidate_clusters:
                for pc in candidate_probe_cls:
                    if pc <= c:
                        for p in candidate_probes:
                            for r in candidate_refines:
                                all_search_space.append([t, b, c, pc, p, r])

    search_X = np.array(all_search_space)
    search_X_scaled = scaler.transform(search_X)

    pred_recs = np.clip(gp_recall.predict(search_X_scaled), 0.0, 1.0)
    pred_times = np.exp(gp_time.predict(search_X_scaled))
    pred_mems = np.exp(gp_mem.predict(search_X_scaled))

    scenarios = {}

    # 1. memory scenario: Minimize index memory, subject to predicted Recall >= 95%
    idx_mem = np.where(pred_recs >= 0.95)[0]
    if len(idx_mem) > 0:
        best_mem_idx = idx_mem[np.argmin(pred_mems[idx_mem])]
        best_cfg = search_X[best_mem_idx]
        scenarios['memory'] = {
            'index_params': {'n_tables': int(best_cfg[0]), 'n_bits': int(best_cfg[1]), 'n_clusters': int(best_cfg[2])},
            'query_params': {'n_probe_clusters': int(best_cfg[3]), 'n_probes': int(best_cfg[4]), 'refine_r': int(best_cfg[5])}
        }
        print(f"Memory Scenario (Recall >= 95%%): tables={best_cfg[0]}, bits={best_cfg[1]}, clusters={best_cfg[2]}, probe_cl={best_cfg[3]}, probes={best_cfg[4]}, refine_r={best_cfg[5]}")
    else:
        # Fallback to highest predicted recall
        best_mem_idx = np.argmax(pred_recs)
        best_cfg = search_X[best_mem_idx]
        scenarios['memory'] = {
            'index_params': {'n_tables': int(best_cfg[0]), 'n_bits': int(best_cfg[1]), 'n_clusters': int(best_cfg[2])},
            'query_params': {'n_probe_clusters': int(best_cfg[3]), 'n_probes': int(best_cfg[4]), 'refine_r': int(best_cfg[5])}
        }

    # 2. fast scenario: Minimize query time, subject to predicted Recall >= 80%
    idx_fast = np.where(pred_recs >= 0.80)[0]
    if len(idx_fast) > 0:
        best_fast_idx = idx_fast[np.argmin(pred_times[idx_fast])]
        best_cfg = search_X[best_fast_idx]
        scenarios['fast'] = {
            'index_params': {'n_tables': int(best_cfg[0]), 'n_bits': int(best_cfg[1]), 'n_clusters': int(best_cfg[2])},
            'query_params': {'n_probe_clusters': int(best_cfg[3]), 'n_probes': int(best_cfg[4]), 'refine_r': int(best_cfg[5])}
        }
        print(f"Fast Scenario (Recall >= 80%%): tables={best_cfg[0]}, bits={best_cfg[1]}, clusters={best_cfg[2]}, probe_cl={best_cfg[3]}, probes={best_cfg[4]}, refine_r={best_cfg[5]}")
    else:
        best_fast_idx = np.argmax(pred_recs)
        best_cfg = search_X[best_fast_idx]
        scenarios['fast'] = {
            'index_params': {'n_tables': int(best_cfg[0]), 'n_bits': int(best_cfg[1]), 'n_clusters': int(best_cfg[2])},
            'query_params': {'n_probe_clusters': int(best_cfg[3]), 'n_probes': int(best_cfg[4]), 'refine_r': int(best_cfg[5])}
        }

    # 3. high_recall scenario: Minimize query time, subject to predicted Recall >= 95%
    idx_hr = np.where(pred_recs >= 0.95)[0]
    if len(idx_hr) > 0:
        best_hr_idx = idx_hr[np.argmin(pred_times[idx_hr])]
        best_cfg = search_X[best_hr_idx]
        scenarios['high_recall'] = {
            'index_params': {'n_tables': int(best_cfg[0]), 'n_bits': int(best_cfg[1]), 'n_clusters': int(best_cfg[2])},
            'query_params': {'n_probe_clusters': int(best_cfg[3]), 'n_probes': int(best_cfg[4]), 'refine_r': int(best_cfg[5])}
        }
        print(f"High Recall Scenario (Recall >= 95%%): tables={best_cfg[0]}, bits={best_cfg[1]}, clusters={best_cfg[2]}, probe_cl={best_cfg[3]}, probes={best_cfg[4]}, refine_r={best_cfg[5]}")
    else:
        best_hr_idx = np.argmax(pred_recs)
        best_cfg = search_X[best_hr_idx]
        scenarios['high_recall'] = {
            'index_params': {'n_tables': int(best_cfg[0]), 'n_bits': int(best_cfg[1]), 'n_clusters': int(best_cfg[2])},
            'query_params': {'n_probe_clusters': int(best_cfg[3]), 'n_probes': int(best_cfg[4]), 'refine_r': int(best_cfg[5])}
        }

    # Write tuned scenarios.yaml
    output_path = "../competitors/ivf_lsh/scenarios.yaml"
    print(f"\nWriting tuned scenarios to {output_path}...")
    
    yaml_structure = {
        'scenarios': {
            'memory': {'default': scenarios['memory']},
            'fast': {'default': scenarios['fast']},
            'high_recall': {'default': scenarios['high_recall']}
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(yaml_structure, f, default_flow_style=False)

    print("✅ Parameter tuning complete and scenarios.yaml updated successfully!")

if __name__ == "__main__":
    main()
