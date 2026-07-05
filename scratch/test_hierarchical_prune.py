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

n_train = min(100000, len(train))
n_test = min(100, len(test))
train_sub = train[:n_train]
test_sub = test[:n_test]

print("Computing ground truth on first 100 queries...")
linear_index = LinearANN()
linear_index.fit(train_sub)
ground_truth = [linear_index.query(q, 5) for q in test_sub]

def evaluate_config(n_tables, n_bits, n_coarse, n_fine, n_probes, n_probe_coarse, n_probe_fine, dim_coarse, prune_size):
    lsh = LSHANN(n_tables=n_tables, n_bits=n_bits, n_coarse_clusters=n_coarse, n_fine_clusters=n_fine)
    t0 = time.time()
    lsh.fit(train_sub)
    fit_time = time.time() - t0
    
    t0 = time.time()
    results = []
    for q in test_sub:
        res = lsh.query(
            q, k=5, n_probes=n_probes, n_probe_coarse=n_probe_coarse, n_probe_fine=n_probe_fine,
            dim_coarse=dim_coarse, prune_size=prune_size
        )
        results.append(res)
    query_time = (time.time() - t0) * 1000 / n_test # ms per query
    
    # Compute recall
    recalls = []
    for i in range(n_test):
        true_set = set(ground_truth[i][:5])
        pred_set = set(results[i][:5])
        intersection = true_set.intersection(pred_set)
        recalls.append(len(intersection) / 5.0)
    avg_recall = np.mean(recalls)
    
    avg_dists = lsh.total_distances_count() / n_test
    
    return fit_time, query_time, avg_recall, avg_dists

print(f"\n--- Hierarchical IVF-LSH + Pruning Parameter Sweep (N={n_train}, Q={n_test}) ---")
print(f"{'tables':<6} | {'bits':<4} | {'coarse':<6} | {'fine':<4} | {'probes':<6} | {'p_crs':<5} | {'p_fin':<5} | {'d_crs':<5} | {'prune':<5} | {'Recall@5':<10} | {'Query (ms)':<10} | {'Fit (s)':<8} | {'Avg Dists':<8}")
print("-" * 115)

configs = [
    # Baseline: 32 coarse, 16 fine, 20 tables
    (20, 10, 32, 16, 15, 8, 16, 48, 120),
    # Increase probes and prune size
    (20, 10, 32, 16, 15, 12, 32, 64, 200),
    (20, 10, 32, 16, 15, 16, 64, 64, 300),
    (20, 10, 32, 16, 20, 16, 64, 64, 400),
    # Finer tuning: less bits (8 bits instead of 10 bits) for less sparsity
    (20, 8, 32, 16, 10, 12, 32, 64, 300),
    (20, 8, 32, 16, 10, 16, 64, 64, 400),
    (25, 8, 32, 16, 10, 16, 64, 64, 500),
]

for t, b, c_coarse, c_fine, p, p_coarse, p_fine, d_coarse, prune in configs:
    fit_t, q_t, recall, avg_dists = evaluate_config(t, b, c_coarse, c_fine, p, p_coarse, p_fine, d_coarse, prune)
    print(f"{t:<6} | {b:<4} | {c_coarse:<6} | {c_fine:<4} | {p:<6} | {p_coarse:<5} | {p_fine:<5} | {d_coarse:<5} | {prune:<5} | {recall:<10.4f} | {q_t:<10.4f} | {fit_t:<8.4f} | {avg_dists:<8.1f}")
