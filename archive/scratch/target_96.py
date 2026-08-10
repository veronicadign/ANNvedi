import os
import time
import h5py
import numpy as np
from linear_ann.wrapper import LinearANN
from lsh_ann.wrapper import LSHANN
from algorithm import Algorithm

dataset_path = "dataset/yahoo-minilm-public.hdf5"

if not os.path.exists(dataset_path):
    print(f"Dataset not found at {dataset_path}")
    exit(1)

print("Loading full dataset...")
with h5py.File(dataset_path, 'r') as f:
    train = f['train'][:]
    test = f['test'][:]

n_train = min(100000, len(train))
train_sub = train[:n_train]
print("Computing ground truth on first 100 queries...")
linear_index = LinearANN()
linear_index.fit(train_sub)
ground_truth = [linear_index.query(q, 5) for q in test[:100]]

def evaluate_config(n_tables, n_bits, n_clusters, n_probe_clusters, n_probes):
    lsh = LSHANN(n_tables=n_tables, n_bits=n_bits, n_clusters=n_clusters)
    t0 = time.time()
    lsh.fit(train_sub)
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
    
    return fit_time, query_time, avg_recall

# Configurations to test to reach exactly >= 96% recall
configs = [
    # Baseline fast was: 20, 10, 256, 16, 10 -> 94.58% recall
    (20, 10, 256, 20, 10),  # Increase probe clusters
    (20, 10, 256, 24, 10),  # Increase probe clusters more
    (20, 10, 256, 16, 15),  # Increase LSH multi-probe
    (20, 10, 256, 20, 15),  # Increase both
    (25, 10, 256, 16, 10),  # Increase tables to 25
    (25, 10, 256, 20, 10),  # Increase tables + probe clusters
]

print(f"\n--- Testing Configurations targeting >= 96% Recall ---")
print(f"{'n_tables':<10} | {'n_bits':<8} | {'clusters':<10} | {'probes_cl':<10} | {'n_probes':<8} | {'Recall@5':<10} | {'Query (ms)':<12} | {'Fit (s)':<10}")
print("-" * 90)

for n_tables, n_bits, n_clusters, n_probe_clusters, n_probes in configs:
    fit_t, q_t, recall = evaluate_config(n_tables, n_bits, n_clusters, n_probe_clusters, n_probes)
    print(f"{n_tables:<10} | {n_bits:<8} | {n_clusters:<10} | {n_probe_clusters:<10} | {n_probes:<8} | {recall:<10.4f} | {q_t:<12.4f} | {fit_t:<10.4f}")
