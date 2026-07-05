import json
import os

results_file = "scratch/dense_sweep_results.json"
if not os.path.exists(results_file):
    print("No sweep results file found!")
    exit(1)

with open(results_file, 'r') as f:
    results = json.load(f)

print(f"Total evaluated configurations: {len(results)}")

# Filter and sort
valid = [r for r in results if r['recall'] >= 0.85]
valid.sort(key=lambda x: x['query_time_ms'])

print(f"\n=== TOP 15 SWEEP CONFIGURATIONS (Recall >= 85%, Sorted by Latency) ===")
print(f"{'Rank':<4} | {'tables':<6} | {'bits':<4} | {'clusters':<8} | {'probe_cl':<8} | {'probes':<6} | {'Recall':<10} | {'Query (ms)':<10} | {'Fit (s)':<8}")
print("-" * 80)
for idx, r in enumerate(valid[:15]):
    print(f"{idx+1:<4} | {r['n_tables']:<6} | {r['n_bits']:<4} | {r['n_clusters']:<8} | {r['n_probe_clusters']:<8} | {r['n_probes']:<6} | {r['recall']:<10.4f} | {r['query_time_ms']:<10.4f} | {r['fit_time_s']:<8.2f}")
