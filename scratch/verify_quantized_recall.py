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

# Evaluate optimal 95% target config with new SQ8 + Refine
lsh = LSHANN(n_tables=20, n_bits=10, n_clusters=256)
lsh.fit(train)

results = []
t0 = time.time()
for q in test[:100]:
    res = lsh.query(q, k=5, n_probes=10, n_probe_clusters=16)
    results.append(res)
query_time = (time.time() - t0) * 1000 / 100

recalls = []
for i in range(100):
    true_set = set(ground_truth[i][:5])
    pred_set = set(results[i][:5])
    intersection = true_set.intersection(pred_set)
    recalls.append(len(intersection) / 5.0)
avg_recall = np.mean(recalls)

print(f"\n=== SQ8 + Float Refinement Results (Full Dataset) ===")
print(f"Recall@5      : {avg_recall:.4f}")
print(f"Query Latency : {query_time:.4f} ms")
