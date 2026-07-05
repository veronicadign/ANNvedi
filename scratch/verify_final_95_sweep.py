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

print("Loading full dataset...")
with h5py.File(dataset_path, 'r') as f:
    train = f['train'][:]
    test = f['test'][:]

print("Computing ground truth on first 100 queries...")
linear_index = LinearANN()
linear_index.fit(train)
ground_truth = [linear_index.query(q, 5) for q in test[:100]]

def evaluate_config(n_tables, n_bits, n_clusters, n_probe_clusters, n_probes):
    lsh = LSHANN(n_tables=n_tables, n_bits=n_bits, n_clusters=n_clusters)
    t0 = time.time()
    lsh.fit(train)
    fit_time = time.time() - t0
    
    t0 = time.time()
    results = []
    for q in test[:100]:
        res = lsh.query(q, k=5, n_probes=n_probes, n_probe_clusters=n_probe_clusters)
        results.append(res)
    query_time = (time.time() - t0) * 1000 / 100 # ms per query
    
    # Compute recall
    recalls = []
    for i in range(100):
        true_set = set(ground_truth[i][:5])
        pred_set = set(results[i][:5])
        intersection = true_set.intersection(pred_set)
        recalls.append(len(intersection) / 5.0)
    avg_recall = np.mean(recalls)
    
    avg_dists = lsh.total_distances_count() / 100
    
    return fit_time, query_time, avg_recall, avg_dists

# Target candidate configurations
configs = [
    # Candidate 1: 20 tables, 9 bits (Very fast, high quality)
    (20, 9, 256, 18, 8),
    # Candidate 2: 15 tables, 8 bits (Very low build time, fewer tables)
    (15, 8, 256, 20, 8),
    # Candidate 3: 20 tables, 8 bits (Alternative)
    (20, 8, 256, 18, 6),
]

print(f"\n--- Verifying Optimized Sweep Candidates on Full Dataset ---")
print(f"{'tables':<6} | {'bits':<4} | {'clusters':<10} | {'probe_cl':<8} | {'probes':<6} | {'Recall@5':<10} | {'Query (ms)':<12} | {'Fit (s)':<10} | {'Avg Dists':<10}")
print("-" * 100)

for t, b, c, pc, p in configs:
    fit_t, q_t, recall, avg_dists = evaluate_config(t, b, c, pc, p)
    print(f"{t:<6} | {b:<4} | {c:<10} | {pc:<8} | {p:<6} | {recall:<10.4f} | {q_t:<12.4f} | {fit_t:<10.4f} | {avg_dists:<10.1f}")
