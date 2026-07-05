import os
import time
import h5py
import numpy as np
import yaml
from concurrent.futures import ThreadPoolExecutor
from algorithm import Algorithm

dataset_path = "dataset/yahoo-minilm-public.hdf5"

if not os.path.exists(dataset_path):
    print(f"Dataset not found at {dataset_path}")
    exit(1)

print("Loading full dataset...")
with h5py.File(dataset_path, 'r') as f:
    train = f['train'][:]
    test = f['test'][:]

print(f"Full Train shape: {train.shape}, dtype: {train.dtype}")
print(f"Full Test shape: {test.shape}, dtype: {test.dtype}")

# Load scenarios
with open("scenarios.yaml", "r") as f:
    scenarios_data = yaml.safe_load(f)["scenarios"]

# First compute the true ground truth on all test queries using brute-force Linear
print("\nComputing linear search ground truth on full dataset...")
linear_algo = Algorithm()
t0 = time.time()
linear_algo.fit(train)
fit_time_linear = time.time() - t0

t0 = time.time()
ground_truth = []
for q in test:
    ground_truth.append(linear_algo.query(q, 5))
query_time_linear = (time.time() - t0) * 1000 / len(test)

print(f"Linear Fit Time: {fit_time_linear:.4f} s")
print(f"Linear Avg Query Time (Sequential): {query_time_linear:.4f} ms")

print("\n--- Evaluating Configured Scenarios (Sequential vs Parallel) ---")
print(f"{'Scenario':<15} | {'Recall@5':<8} | {'Seq Avg (ms)':<12} | {'Par Avg (ms)':<12} | {'Speedup':<8} | {'Par QPS':<10} | {'Fit (s)':<8} | {'Dist Computes':<12}")
print("-" * 105)

for scenario_name, scenario_cfg in scenarios_data.items():
    cfg = scenario_cfg["default"]
    index_params = cfg.get("index_params", {})
    query_params = cfg.get("query_params", {})
    
    # Initialize algorithm
    algo = Algorithm()
    
    # Measure fit time
    t0 = time.time()
    algo.fit(train, **index_params)
    fit_time = time.time() - t0
    
    # 1. Sequential query loop
    t0 = time.time()
    results_seq = []
    for q in test:
        res = algo.query(q, 5, **query_params)
        results_seq.append(res)
    seq_time_total = time.time() - t0
    seq_query_avg_ms = (seq_time_total * 1000) / len(test)
    
    # 2. Parallel query loop using ThreadPoolExecutor
    num_threads = os.cpu_count() or 4
    
    # Helper query worker
    def query_worker(q):
        return algo.query(q, 5, **query_params)
    
    # Run parallel
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results_par = list(executor.map(query_worker, test))
    par_time_total = time.time() - t0
    par_query_avg_ms = (par_time_total * 1000) / len(test)
    
    # QPS
    par_qps = len(test) / par_time_total
    
    # Speedup
    speedup = seq_time_total / par_time_total
    
    # Calculate recall using parallel results (ensure they match ground truth)
    recalls = []
    for i in range(len(test)):
        true_set = set(ground_truth[i][:5])
        pred_set = set(results_par[i][:5])
        intersection = true_set.intersection(pred_set)
        recalls.append(len(intersection) / 5.0)
    avg_recall = np.mean(recalls)
    
    # Sanity check: verify sequential and parallel results match
    mismatches = 0
    for i in range(len(test)):
        if not np.array_equal(results_seq[i], results_par[i]):
            mismatches += 1
    if mismatches > 0:
        print(f"\n[WARNING] Scenario {scenario_name}: Parallel and sequential results had {mismatches} mismatches! Check thread-safety.")
    
    dist_count = algo.get_n_distances()
    
    print(f"{scenario_name:<15} | {avg_recall:<8.4f} | {seq_query_avg_ms:<12.4f} | {par_query_avg_ms:<12.4f} | {speedup:<8.2f}x | {par_qps:<10.1f} | {fit_time:<8.4f} | {dist_count:<12}")

print("\nEvaluation complete.")
