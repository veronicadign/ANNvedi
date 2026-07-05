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

def evaluate_config(n_tables, n_bits, n_clusters, n_probe_clusters, n_probes):
    try:
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
            true_set = set(ground_truth[i][:5])
            pred_set = set(results[i][:5])
            intersection = true_set.intersection(pred_set)
            recalls.append(len(intersection) / 5.0)
        avg_recall = np.mean(recalls)
        
        return fit_time, query_time, avg_recall
    except Exception as e:
        return 100.0, 100.0, 0.0

# Sweep grid
configs = [
    # Tune 15 tables with 8 or 9 bits
    (15, 8, 256, 20, 8),
    (15, 8, 256, 24, 8),
    (15, 9, 256, 20, 10),
    (15, 9, 256, 24, 10),
    # Tune 20 tables with 8 or 9 bits
    (20, 8, 256, 18, 6),
    (20, 8, 256, 20, 6),
    (20, 9, 256, 18, 8),
    (20, 9, 256, 20, 8),
]

print(f"\n--- Sweeping tables & bits for Recall >= 95% (N=100k, Q=100) ---")
print(f"{'tables':<6} | {'bits':<4} | {'clusters':<10} | {'probe_cl':<8} | {'probes':<6} | {'Recall@5':<10} | {'Query (ms)':<12} | {'Fit (s)':<10}")
print("-" * 80)

results_list = []
for t, b, c, pc, p in configs:
    fit_t, q_t, recall = evaluate_config(t, b, c, pc, p)
    results_list.append((t, b, c, pc, p, recall, q_t, fit_t))
    print(f"{t:<6} | {b:<4} | {c:<10} | {pc:<8} | {p:<6} | {recall:<10.4f} | {q_t:<12.4f} | {fit_t:<10.4f}")

# Find configurations that achieve >= 95% recall, sorted by query time
print("\n--- Configurations achieving >= 95% Recall (Sorted by Speed) ---")
valid_configs = [res for res in results_list if res[5] >= 0.95]
valid_configs.sort(key=lambda x: x[6])

for idx, (t, b, c, pc, p, recall, q_t, fit_t) in enumerate(valid_configs):
    print(f"Rank {idx+1}: tables={t}, bits={b}, clusters={c}, probe_cl={pc}, probes={p} | Recall={recall:.4f} | Query={q_t:.2f} ms | Fit={fit_t:.2f} s")
