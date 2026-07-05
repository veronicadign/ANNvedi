import os
import time
import h5py
import numpy as np
from linear_ann.wrapper import LinearANN
from lsh_ann.wrapper import LSHANN

dataset_path = "dataset/yahoo-minilm-public.hdf5"

if not os.path.exists(dataset_path):
    print(f"Dataset not found at {dataset_path}")
    exit(1)

print("Loading dataset...")
with h5py.File(dataset_path, 'r') as f:
    train = f['train'][:]
    test = f['test'][:]

# We'll evaluate on 100k points and 100 queries for a quick evaluation,
# and also show full dataset scaling if needed.
n_train = min(100000, len(train))
n_test = min(100, len(test))
train_sub = train[:n_train]
test_sub = test[:n_test]

# 1. Get ground truth
linear_index = LinearANN()
linear_index.fit(train_sub)
linear_results = [linear_index.query(q, 5) for q in test_sub]

# Helper to run evaluations
def evaluate_ivf_lsh(n_tables, n_bits, n_clusters, n_probe_clusters, n_probes):
    lsh = LSHANN(n_tables=n_tables, n_bits=n_bits, n_clusters=n_clusters)
    t0 = time.time()
    lsh.fit(train_sub)
    fit_time = time.time() - t0
    
    t0 = time.time()
    results = []
    for q in test_sub:
        res = lsh.query(q, k=5, n_probes=n_probes, n_probe_clusters=n_probe_clusters)
        results.append(res)
    query_time = (time.time() - t0) * 1000 / n_test # ms per query
    
    # Compute recall
    recalls = []
    for i in range(n_test):
        true_set = set(linear_results[i][:5])
        pred_set = set(results[i][:5])
        intersection = true_set.intersection(pred_set)
        recalls.append(len(intersection) / 5.0)
    avg_recall = np.mean(recalls)
    
    dist_count = lsh.total_distances_count()
    avg_dists = dist_count / n_test
    
    return fit_time, query_time, avg_recall, avg_dists

print(f"\n--- IVF-LSH Parameter Evaluation (N={n_train}, Q={n_test}) ---")
print(f"{'n_tables':<10} | {'n_bits':<8} | {'clusters':<10} | {'probes_cl':<10} | {'n_probes':<8} | {'Recall@5':<10} | {'Query (ms)':<12} | {'Fit (s)':<10} | {'Avg Dists':<10}")
print("-" * 105)

# Test various configurations
configs = [
    # Fixed clusters, vary probed clusters
    (10, 8, 256, 4, 0),
    (10, 8, 256, 8, 0),
    (10, 8, 256, 16, 0),
    # Fixed clusters, vary probed clusters + LSH multi-probe
    (10, 8, 256, 8, 5),
    (10, 8, 256, 16, 5),
    (10, 8, 256, 32, 5),
    # Varying cluster counts
    (10, 8, 512, 16, 5),
    (10, 8, 512, 32, 5),
    (10, 8, 1024, 32, 5),
    # Higher quality setups
    (20, 10, 256, 16, 10),
    (20, 10, 512, 32, 10),
    (20, 10, 512, 64, 10),
]

for n_tables, n_bits, n_clusters, n_probe_clusters, n_probes in configs:
    fit_time, query_time, recall, avg_dists = evaluate_ivf_lsh(n_tables, n_bits, n_clusters, n_probe_clusters, n_probes)
    print(f"{n_tables:<10} | {n_bits:<8} | {n_clusters:<10} | {n_probe_clusters:<10} | {n_probes:<8} | {recall:<10.4f} | {query_time:<12.4f} | {fit_time:<10.4f} | {avg_dists:<10.1f}")
