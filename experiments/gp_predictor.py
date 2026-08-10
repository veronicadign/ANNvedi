import os
import time
import h5py
import numpy as np
import random
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from linear_ann.wrapper import LinearANN
from lsh_ann.wrapper import LSHANN

# Suppress convergence warnings from GP fitting
warnings.filterwarnings('ignore')

dataset_path = "dataset/yahoo-minilm-public.hdf5"

if not os.path.exists(dataset_path):
    print(f"Dataset not found at {dataset_path}")
    exit(1)

print("Loading dataset...")
with h5py.File(dataset_path, 'r') as f:
    train = f['train'][:]
    test = f['test'][:]

# Use subsample for fast data collection
n_train = min(100000, len(train))
n_test = min(50, len(test))
train_sub = train[:n_train]
test_sub = test[:n_test]

print("Computing ground truth for recall evaluation...")
linear_index = LinearANN()
linear_index.fit(train_sub)
linear_results = [linear_index.query(q, 5) for q in test_sub]

def measure_config(n_tables, n_bits, n_clusters, n_probe_clusters, n_probes):
    try:
        lsh = LSHANN(n_tables=n_tables, n_bits=n_bits, n_clusters=n_clusters)
        lsh.fit(train_sub)
        
        t0 = time.time()
        results = []
        for q in test_sub:
            res = lsh.query(q, k=5, n_probes=n_probes, n_probe_clusters=n_probe_clusters)
            results.append(res)
        query_time = (time.time() - t0) * 1000 / n_test # ms per query
        
        recalls = []
        for i in range(n_test):
            true_set = set(linear_results[i][:5])
            pred_set = set(results[i][:5])
            intersection = true_set.intersection(pred_set)
            recalls.append(len(intersection) / 5.0)
        avg_recall = np.mean(recalls)
        
        return avg_recall, query_time
    except Exception as e:
        print(f"Error evaluating config ({n_tables}, {n_bits}, {n_clusters}, {n_probe_clusters}, {n_probes}): {e}")
        return 0.0, 100.0

# Generate unique random configurations
np.random.seed(42)
random.seed(42)

configs = []
while len(configs) < 30:
    n_tables = int(random.choice([5, 10, 15, 20, 30]))
    n_bits = int(random.choice([6, 8, 10, 12]))
    n_clusters = int(random.choice([64, 128, 256, 512]))
    n_probe_clusters = int(random.choice([4, 8, 16, 32]))
    n_probes = int(random.choice([0, 5, 10, 20, 30]))
    
    # Ensure n_probe_clusters <= n_clusters
    if n_probe_clusters > n_clusters:
        n_probe_clusters = n_clusters
        
    cfg = (n_tables, n_bits, n_clusters, n_probe_clusters, n_probes)
    if cfg not in configs:
        configs.append(cfg)

X_data = []
y_recall = []
y_time = []

# Collect training data
print("\nCollecting observations to train Gaussian Processes...")
for idx, (n_tables, n_bits, n_clusters, n_probe_clusters, n_probes) in enumerate(configs):
    print(f"Config {idx+1}/30: tables={n_tables}, bits={n_bits}, clusters={n_clusters}, probe_cl={n_probe_clusters}, probes={n_probes} ... ", end="", flush=True)
    recall, q_time = measure_config(n_tables, n_bits, n_clusters, n_probe_clusters, n_probes)
    print(f"Recall={recall:.4f}, Time={q_time:.2f} ms")
    
    X_data.append([n_tables, n_bits, n_clusters, n_probe_clusters, n_probes])
    X_scaled_single = [n_tables, n_bits, n_clusters, n_probe_clusters, n_probes]
    y_recall.append(recall)
    y_time.append(q_time)

X = np.array(X_data)
y_rec = np.array(y_recall)
# Log transform time to predict positive values stably
y_t = np.log(np.array(y_time))

# Scale input features to zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFitting Gaussian Process Regressors on scaled features...")

# ARD kernel: separate length scale for each scaled parameter
kernel_rec = Matern(length_scale=[1.0]*5, nu=2.5) + WhiteKernel(noise_level=1e-4)
kernel_time = Matern(length_scale=[1.0]*5, nu=2.5) + WhiteKernel(noise_level=1e-3)

gp_recall = GaussianProcessRegressor(kernel=kernel_rec, n_restarts_optimizer=10, random_state=42)
gp_recall.fit(X_scaled, y_rec)

gp_time = GaussianProcessRegressor(kernel=kernel_time, n_restarts_optimizer=10, random_state=42)
gp_time.fit(X_scaled, y_t)

print("Gaussian Processes trained successfully.")

# Validate on 5 unseen test configurations
test_configs = [
    (10, 8, 256, 16, 5),
    (20, 10, 256, 8, 10),
    (5, 10, 128, 8, 5),
    (15, 8, 512, 16, 15),
    (25, 12, 256, 32, 10)
]

print("\n--- Validation on Unseen Test Configurations ---")
for n_tables, n_bits, n_clusters, n_probe_clusters, n_probes in test_configs:
    # Actual measurement
    act_rec, act_t = measure_config(n_tables, n_bits, n_clusters, n_probe_clusters, n_probes)
    
    # Scale input and Predict using GP
    test_x = np.array([[n_tables, n_bits, n_clusters, n_probe_clusters, n_probes]])
    test_x_scaled = scaler.transform(test_x)
    
    pred_rec, std_rec = gp_recall.predict(test_x_scaled, return_std=True)
    pred_log_t, std_log_t = gp_time.predict(test_x_scaled, return_std=True)
    
    # Exponential back to time domain
    pred_t = np.exp(pred_log_t[0])
    
    # Clip recall prediction to [0.0, 1.0]
    pred_rec = np.clip(pred_rec[0], 0.0, 1.0)
    
    print(f"Config: tables={n_tables}, bits={n_bits}, clusters={n_clusters}, probe_cl={n_probe_clusters}, probes={n_probes}")
    print(f"  Recall: Actual={act_rec:.4f} | Predicted={pred_rec:.4f} (std={std_rec[0]:.4f})")
    print(f"  Time  : Actual={act_t:.2f} ms | Predicted={pred_t:.2f} ms (std_log={std_log_t[0]:.4f})")
    print()

# Search for optimal configurations over a dense parameter grid using GP predictions
print("\n--- Finding Optimal Configurations Using GP Predictions ---")

candidate_tables = [5, 10, 15, 20, 30]
candidate_bits = [6, 8, 10, 12]
candidate_clusters = [64, 128, 256, 512]
candidate_probe_cls = [4, 8, 16, 32]
candidate_probes = [0, 5, 10, 15, 20, 30]

all_search_space = []
for t in candidate_tables:
    for b in candidate_bits:
        for c in candidate_clusters:
            for pc in candidate_probe_cls:
                if pc <= c:
                    for p in candidate_probes:
                        all_search_space.append([t, b, c, pc, p])

search_X = np.array(all_search_space)
search_X_scaled = scaler.transform(search_X)

pred_recs = np.clip(gp_recall.predict(search_X_scaled), 0.0, 1.0)
pred_times = np.exp(gp_time.predict(search_X_scaled))

# Optimal for Recall >= 90%
idx_90 = np.where(pred_recs >= 0.90)[0]
if len(idx_90) > 0:
    best_90_idx = idx_90[np.argmin(pred_times[idx_90])]
    best_cfg = search_X[best_90_idx]
    print(f"Best Predicted Configuration for Recall >= 90%:")
    print(f"  Parameters: tables={best_cfg[0]}, bits={best_cfg[1]}, clusters={best_cfg[2]}, probe_cl={best_cfg[3]}, probes={best_cfg[4]}")
    print(f"  Predicted Recall: {pred_recs[best_90_idx]:.4f} | Predicted Time: {pred_times[best_90_idx]:.2f} ms")
else:
    print("No configuration predicted to achieve >= 90% recall.")

# Optimal for Recall >= 95%
idx_95 = np.where(pred_recs >= 0.95)[0]
if len(idx_95) > 0:
    best_95_idx = idx_95[np.argmin(pred_times[idx_95])]
    best_cfg = search_X[best_95_idx]
    print(f"\nBest Predicted Configuration for Recall >= 95%:")
    print(f"  Parameters: tables={best_cfg[0]}, bits={best_cfg[1]}, clusters={best_cfg[2]}, probe_cl={best_cfg[3]}, probes={best_cfg[4]}")
    print(f"  Predicted Recall: {pred_recs[best_95_idx]:.4f} | Predicted Time: {pred_times[best_95_idx]:.2f} ms")
else:
    print("\nNo configuration predicted to achieve >= 95% recall.")
