from python.wrapper import LinearANN

import numpy as np
import h5py
import os

output_dir = "/content/datasets"
hdf5_files = [f for f in os.listdir(output_dir) if f.endswith('.hdf5')]
if hdf5_files:
    first = os.path.join(output_dir, hdf5_files[0])
    print(f"Elaborazione: {first}")
    with h5py.File(first, 'r') as f:
        train = f['/train'][:]
        queries = f['/test'][:5]   # prime 5 query per test rapido

    ann = LinearANN()
    ann.fit(train)

    for i, q in enumerate(queries):
        ids = ann.query(q, 100)
        print(f"Query {i}: primi 5 ID = {ids[:5]}")

    print("Distanze totali calcolate:", ann.total_distances_count())
else:
    print("Nessun dataset trovato in /content/datasets. Prima esegui lo scaricamento con gdown.")