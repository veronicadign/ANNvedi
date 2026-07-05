import os
import time
import h5py
import numpy as np
from HNSW.algorithm import Algorithm
from linear_ann.wrapper import LinearANN

dataset_path = "dataset/yahoo-minilm-public.hdf5"

if not os.path.exists(dataset_path):
    print(f"Dataset not found at {dataset_path}")
    exit(1)

print("Loading full dataset...")
with h5py.File(dataset_path, 'r') as f:
    train = f['train'][:]
    test = f['test'][:]

# Use 100k subset for fast iteration in profiling
subset_size = 100000
train_sub = train[:subset_size]
print(f"Using a subset of {subset_size} vectors for fast profiling.")

print("Computing ground truth on first 100 queries...")
linear_index = LinearANN()
linear_index.fit(train_sub)
ground_truth = [linear_index.query(q, 5) for q in test[:100]]

results_summary = []

for mode in ["float", "sq8", "lsh"]:
    print(f"\n--- Profiling HNSW ({mode} mode) ---")
    
    # 1. Fit profiling
    algo = Algorithm()
    t0 = time.time()
    algo.fit(train_sub, M=16, ef_construction=100, mode=mode)
    fit_time = time.time() - t0
    print(f"HNSW ({mode}) Fit Time: {fit_time:.4f} s")
    
    # 2. Query profiling
    def run_queries(ef_val):
        results = []
        for q in test[:100]:
            res = algo.query(q, k=5, ef=ef_val)
            results.append(res)
        return results

    for ef in [40, 120, 250]:
        algo.get_n_distances() # reset/dummy read if needed, wait, we want cumulative count for queries
        # Reset count in index
        algo._index.reset_distances_count()
        
        t0 = time.time()
        results = run_queries(ef)
        query_time = (time.time() - t0) * 1000 / 100 # ms per query
        
        # Compute recall
        recalls = []
        for i in range(100):
            true_set = set(ground_truth[i][:5])
            pred_set = set(results[i][:5])
            intersection = true_set.intersection(pred_set)
            recalls.append(len(intersection) / 5.0)
        avg_recall = np.mean(recalls)
        avg_dists = algo.get_n_distances() / 100
        
        print(f"HNSW ({mode}, ef={ef:<3}): Recall@5 = {avg_recall:.4f} | Avg Query Time = {query_time:.4f} ms | Avg Distance Computations = {avg_dists:.1f}")
        
        results_summary.append({
            "mode": mode,
            "ef": ef,
            "fit_time": fit_time,
            "query_time": query_time,
            "recall": avg_recall,
            "dists": avg_dists
        })

print("\n\n### unified profiling comparison")
print(f"| {'Mode':<10} | {'ef':<5} | {'Fit Time (s)':<12} | {'Query Time (ms)':<15} | {'Recall@5':<10} | {'Distance Computations':<22} |")
print(f"| :--- | :--- | :--- | :--- | :--- | :--- |")
for r in results_summary:
    print(f"| {r['mode']:<10} | {r['ef']:<5} | {r['fit_time']:<12.2f} | {r['query_time']:<15.4f} | {r['recall']:<10.4f} | {r['dists']:<22.1f} |")
