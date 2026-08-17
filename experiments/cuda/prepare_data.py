#!/usr/bin/env python3
"""Export a dataset into the raw binary layout gpu_filter.cu reads.

Usage: python3 prepare_data.py ../../dataset/yahoo-minilm-public.hdf5 /tmp/gf
"""
import os
import sys

import h5py
import numpy as np

K = 100

def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    with h5py.File(src, "r") as f:
        train = np.asarray(f["train"][:], dtype=np.float32)
        queries = np.asarray(f["test"][:], dtype=np.float32)
        gt = np.asarray(f["distances"][:, :K], dtype=np.float32)

    n, dim = train.shape
    # per-dimension SQ8, identical to hnsw sq8pd
    mn, mx = train.min(axis=0), train.max(axis=0)
    scale = np.where(mx - mn > 1e-8, (mx - mn) / 255.0, 1.0).astype(np.float32)
    offset = np.where(mx - mn > 1e-8, mn, mn).astype(np.float32)
    codes = np.clip(np.round((train - offset) / scale), 0, 255).astype(np.uint8)

    with open(f"{out}/meta.txt", "w") as f:
        f.write(f"{n} {dim} {queries.shape[0]} {K}\n")
    codes.tofile(f"{out}/codes.u8")
    scale.tofile(f"{out}/scale.f32")
    offset.tofile(f"{out}/offset.f32")
    train.tofile(f"{out}/train.f32")
    queries.tofile(f"{out}/queries.f32")
    gt.tofile(f"{out}/gt.f32")
    print(f"wrote {out}: N={n} dim={dim} queries={queries.shape[0]} k={K} "
          f"(codes {codes.nbytes>>20} MB, train {train.nbytes>>20} MB)")

if __name__ == "__main__":
    main()
