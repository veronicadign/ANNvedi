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

n_train_sub = 100000
train_sub = train[:n_train_sub]
test_sub = test[:100]

print("Computing ground truth on first 100 queries...")
linear_index = LinearANN()
linear_index.fit(train_sub)
ground_truth = [linear_index.query(q, 5) for q in test_sub]

# Dynamic dense parameter search space
n_tables_opts = [10, 15, 20, 25]
n_bits_opts = [8, 9, 10]
n_clusters_opts = [128, 256]
n_probe_clusters_opts = [8, 12, 16, 20]
n_probes_opts = [4, 6, 8, 10, 12]

print(f"\nStarting comprehensive grid search over LSH parameter space...")
results = []
count = 0
total_configs = len(n_tables_opts) * len(n_bits_opts) * len(n_clusters_opts) * len(n_probe_clusters_opts) * len(n_probes_opts)
print(f"Total configurations to evaluate: {total_configs}")

for n_tables in n_tables_opts:
    for n_bits in n_bits_opts:
        for n_clusters in n_clusters_opts:
            # Fit LSH Index once for this index layout configuration
            try:
                lsh = LSHANN(n_tables=n_tables, n_bits=n_bits, n_clusters=n_clusters)
                t0 = time.time()
                lsh.fit(train_sub)
                fit_time = time.time() - t0
            except Exception as e:
                continue

            for n_probe_clusters in n_probe_clusters_opts:
                if n_probe_clusters > n_clusters:
                    continue
                for n_probes in n_probes_opts:
                    count += 1
                    t0 = time.time()
                    preds = []
                    for q in test_sub:
                        res = lsh.query(q, k=5, n_probes=n_probes, n_probe_clusters=n_probe_clusters)
                        preds.append(res)
                    query_time = (time.time() - t0) * 1000 / len(test_sub)

                    # Compute Recall
                    recalls = []
                    for idx in range(len(test_sub)):
                        true_set = set(ground_truth[idx][:5])
                        pred_set = set(preds[idx][:5])
                        intersection = true_set.intersection(pred_set)
                        recalls.append(len(intersection) / 5.0)
                    recall = np.mean(recalls)

                    results.append({
                        'n_tables': n_tables,
                        'n_bits': n_bits,
                        'n_clusters': n_clusters,
                        'n_probe_clusters': n_probe_clusters,
                        'n_probes': n_probes,
                        'recall': recall,
                        'query_time_ms': query_time,
                        'fit_time_s': fit_time
                    })

                    if count % 20 == 0:
                        print(f"Evaluated {count}/{total_configs} configurations...")

print("\nGrid search completed.")

# Filter configurations with recall >= 95%
valid = [r for r in results if r['recall'] >= 0.95]
valid.sort(key=lambda x: x['query_time_ms'])

print(f"\n=== Top 10 Configurations achieving >= 95% Recall (Sorted by Speed) ===")
for rank, r in enumerate(valid[:10]):
    print(f"Rank {rank+1:2d}: tables={r['n_tables']:2d}, bits={r['n_bits']:2d}, clusters={r['n_clusters']:3d}, probe_cl={r['n_probe_clusters']:2d}, probes={r['n_probes']:2d} | Recall={r['recall']:.4f} | Query={r['query_time_ms']:.4f} ms | Fit={r['fit_time_s']:.2f} s")

# Choose top 3 configurations to evaluate on the FULL dataset
print("\n--- Evaluating Top 3 Configurations on the FULL Dataset ---")
top_3 = valid[:3]

# Compute ground truth on full dataset
print("Computing Ground Truth on full dataset for 100 queries...")
linear_full = LinearANN()
linear_full.fit(train)
ground_truth_full = [linear_full.query(q, 5) for q in test[:100]]

print(f"\n{'tables':<6} | {'bits':<4} | {'clusters':<8} | {'probe_cl':<8} | {'probes':<6} | {'Recall@5':<10} | {'Query (ms)':<12} | {'Fit (s)':<10}")
print("-" * 85)

for r in top_3:
    lsh_full = LSHANN(n_tables=r['n_tables'], n_bits=r['n_bits'], n_clusters=r['n_clusters'])
    
    t0 = time.time()
    lsh_full.fit(train)
    fit_full_time = time.time() - t0
    
    t0 = time.time()
    preds_full = []
    for q in test[:100]:
        res = lsh_full.query(q, k=5, n_probes=r['n_probes'], n_probe_clusters=r['n_probe_clusters'])
        preds_full.append(res)
    query_full_time = (time.time() - t0) * 1000 / 100
    
    recalls_full = []
    for idx in range(100):
        true_set = set(ground_truth_full[idx][:5])
        pred_set = set(preds_full[idx][:5])
        intersection = true_set.intersection(pred_set)
        recalls_full.append(len(intersection) / 5.0)
    recall_full = np.mean(recalls_full)
    
    print(f"{r['n_tables']:<6} | {r['n_bits']:<4} | {r['n_clusters']:<8} | {r['n_probe_clusters']:<8} | {r['n_probes']:<6} | {recall_full:<10.4f} | {query_full_time:<12.4f} | {fit_full_time:<10.2f}")
