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

print("\n--- Verifying Target Configuration on Full Dataset ---")
lsh = LSHANN(n_tables=20, n_bits=10, n_clusters=256)

t0 = time.time()
lsh.fit(train)
fit_time = time.time() - t0
print(f"Index Fit Time: {fit_time:.4f} s")

# Target Parameters: n_probe_clusters=20, n_probes=15
t0 = time.time()
results = []
for q in test[:100]:
    res = lsh.query(q, k=5, n_probes=15, n_probe_clusters=20)
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

print(f"\nFinal Target Performance:")
print(f"  Recall@5: {avg_recall:.4f}")
print(f"  Query Latency: {query_time:.2f} ms")
print(f"  Distances Computed: {lsh.total_distances_count() / 100:.1f} per query")
