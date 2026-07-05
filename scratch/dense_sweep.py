import os
import time
import json
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

# 100k subset
n_train_sub = 100000
train_sub = train[:n_train_sub]
test_sub = test[:100]

print("Computing ground truth on 100 queries...")
linear_index = LinearANN()
linear_index.fit(train_sub)
ground_truth = [linear_index.query(q, 5) for q in test_sub]

# Comprehensive dense search space with step size 1 (+-1 granularity)
n_tables_range = list(range(8, 26))        # step 1
n_bits_range = list(range(7, 12))         # step 1
n_clusters_range = [128, 256]

n_probe_clusters_range = list(range(4, 25)) # step 1
n_probes_range = list(range(0, 16))        # step 1

results_file = "scratch/dense_sweep_results.json"
results = []

# Load existing results if any to allow resume
if os.path.exists(results_file):
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing sweep results.")
    except Exception:
        results = []

completed_set = {
    (r['n_tables'], r['n_bits'], r['n_clusters'], r['n_probe_clusters'], r['n_probes'])
    for r in results
}

print(f"\nStarting smart pruned dense search over LSH parameter combinations...")
print(f"Targeting Recall >= 0.95")

count = 0
t_start_sweep = time.time()

for n_tables in n_tables_range:
    for n_bits in n_bits_range:
        for n_clusters in n_clusters_range:
            # We will use a grid mapping to prune query parameter search space for this index
            grid_recall = {}
            grid_time = {}
            
            # Populate with already completed results for this index
            for r in results:
                if (r['n_tables'] == n_tables and r['n_bits'] == n_bits and r['n_clusters'] == n_clusters):
                    grid_recall[(r['n_probe_clusters'], r['n_probes'])] = r['recall']
                    grid_time[(r['n_probe_clusters'], r['n_probes'])] = r['query_time_ms']

            # Fit index once for this structure config
            index_fitted = False
            lsh = None
            fit_time = 0.0

            # Scan query parameters
            for n_probe_clusters in n_probe_clusters_range:
                if n_probe_clusters > n_clusters:
                    continue
                for n_probes in n_probes_range:
                    key = (n_probe_clusters, n_probes)
                    if key in grid_recall:
                        continue

                    # Smart Pruning Check 1:
                    # If any already evaluated sub-config with >= n_probe_clusters and >= n_probes
                    # had recall < 0.95, then this smaller config MUST also have recall < 0.95.
                    pruned_by_low_recall = False
                    for (pc, p), rec in grid_recall.items():
                        if pc >= n_probe_clusters and p >= n_probes and rec < 0.95:
                            pruned_by_low_recall = True
                            break
                    
                    if pruned_by_low_recall:
                        grid_recall[key] = 0.0
                        grid_time[key] = 999.0
                        continue

                    # Smart Pruning Check 2:
                    # If any already evaluated sub-config with <= n_probe_clusters and <= n_probes
                    # had recall >= 0.95, we can skip querying since we already found a faster
                    # sub-config that reaches the target recall.
                    pruned_by_high_recall = False
                    for (pc, p), rec in grid_recall.items():
                        if pc <= n_probe_clusters and p <= n_probes and rec >= 0.95:
                            pruned_by_high_recall = True
                            break
                    
                    if pruned_by_high_recall:
                        # Skip evaluating since it is guaranteed to be slower than the existing valid config
                        grid_recall[key] = 1.0
                        grid_time[key] = 999.0
                        continue

                    # Fit index on demand
                    if not index_fitted:
                        try:
                            lsh = LSHANN(n_tables=n_tables, n_bits=n_bits, n_clusters=n_clusters)
                            t0 = time.time()
                            lsh.fit(train_sub)
                            fit_time = time.time() - t0
                            index_fitted = True
                        except Exception as e:
                            print(f"Skipping index fit due to error: tables={n_tables}, bits={n_bits}, clusters={n_clusters} ({e})")
                            break
                    
                    if not index_fitted:
                        break

                    # Evaluate query
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

                    grid_recall[key] = float(recall)
                    grid_time[key] = float(query_time)

                    result_entry = {
                        'n_tables': n_tables,
                        'n_bits': n_bits,
                        'n_clusters': n_clusters,
                        'n_probe_clusters': n_probe_clusters,
                        'n_probes': n_probes,
                        'recall': float(recall),
                        'query_time_ms': float(query_time),
                        'fit_time_s': float(fit_time)
                    }
                    results.append(result_entry)
                    completed_set.add((n_tables, n_bits, n_clusters, n_probe_clusters, n_probes))

                    count += 1
                    if count % 100 == 0:
                        print(f"Evaluated {count} configurations. Total sweep time: {time.time() - t_start_sweep:.1f} s")
                        with open(results_file, 'w') as f:
                            json.dump(results, f)

# Save final results
with open(results_file, 'w') as f:
    json.dump(results, f)

print(f"\nSmart sweep completed. Evaluated {count} new configurations.")

# Filter and sort
valid = [r for r in results if r['recall'] >= 0.95]
valid.sort(key=lambda x: x['query_time_ms'])

print(f"\n=== TOP 15 SWEEP CONFIGURATIONS (Recall >= 95%, Sorted by Latency) ===")
print(f"{'Rank':<4} | {'tables':<6} | {'bits':<4} | {'clusters':<8} | {'probe_cl':<8} | {'probes':<6} | {'Recall':<10} | {'Query (ms)':<10} | {'Fit (s)':<8}")
print("-" * 80)
for idx, r in enumerate(valid[:15]):
    print(f"{idx+1:<4} | {r['n_tables']:<6} | {r['n_bits']:<4} | {r['n_clusters']:<8} | {r['n_probe_clusters']:<8} | {r['n_probes']:<6} | {r['recall']:<10.4f} | {r['query_time_ms']:<10.4f} | {r['fit_time_s']:<8.2f}")
