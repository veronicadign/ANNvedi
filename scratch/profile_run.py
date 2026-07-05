import os
import time
import h5py
import numpy as np
from lsh_ann.wrapper import LSHANN

dataset_path = "dataset/yahoo-minilm-public.hdf5"

if not os.path.exists(dataset_path):
    print(f"Dataset not found at {dataset_path}")
    exit(1)

print("Loading full dataset...")
with h5py.File(dataset_path, 'r') as f:
    train = f['train'][:]
    test = f['test'][:]

print("\n--- Initializing LSH Index (Flat IVF-LSH) ---")
lsh = LSHANN(n_tables=20, n_bits=10, n_clusters=256)
t0 = time.time()
lsh.fit(train)
print(f"Index Fit Time: {time.time() - t0:.4f} s")

# Warm up
print("\nWarming up...")
for q in test[:10]:
    lsh.query(q, k=5, n_probes=10, n_probe_clusters=16)

lsh.reset_profile_results()

print("\nRunning 1,000 queries to collect profile logs...")
t_py_start = time.time()
for q in test:
    lsh.query(q, k=5, n_probes=10, n_probe_clusters=16)
t_py_end = time.time()

avg_py_time = (t_py_end - t_py_start) * 1000 / len(test)
profile = lsh.get_profile_results()

# Print timing breakdown
print("\n=== LATENCY BREAKDOWN PROFILE ===")
print(f"Total Python-level Time per Query: {avg_py_time:.4f} ms")
print("-" * 40)

steps = [
    ("avg_centroid_search_ms", "1. Centroid Distance Checks"),
    ("avg_signature_calc_ms", "2. Signature Calculations"),
    ("avg_bucket_lookup_ms", "3. Bucket Lookups & Gathering"),
    ("avg_distance_calc_ms", "4. Candidate distance calculations"),
    ("avg_candidate_sort_ms", "5. Sorting to get top K"),
]

cpp_total = 0.0
for key, label in steps:
    val = profile.get(key, 0.0)
    cpp_total += val
    pct = (val / avg_py_time) * 100 if avg_py_time > 0 else 0
    print(f"{label:<30} : {val:.4f} ms ({pct:.1f}%)")

py_overhead = avg_py_time - cpp_total
pct_overhead = (py_overhead / avg_py_time) * 100 if avg_py_time > 0 else 0
print(f"{'6. Python & pybind11 overhead':<30} : {py_overhead:.4f} ms ({pct_overhead:.1f}%)")
print("-" * 40)
print(f"Total C++ execution time per Query : {cpp_total:.4f} ms ({cpp_total/avg_py_time*100:.1f}%)")
print(f"Avg candidates checked per query  : {lsh.total_distances_count()/len(test):.1f}")
