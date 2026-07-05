#!/bin/env python3
import h5py
import numpy as np
import sys
from pathlib import Path
import urllib.request
from urllib.parse import urlparse
import logging
from icecream import ic
from joblib import Parallel, delayed


logging.basicConfig(level=logging.INFO)

DATA_DIR = Path("datasets")

DATASET_URLS = {
    "landmark-nomic": "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/landmark-nomic-768-normalized.hdf5?download=true",
    "imagenet-clip": "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/imagenet-clip-512-normalized.hdf5?download=true",
    "simplewiki-openai": "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/simplewiki-openai-3072-normalized.hdf5?download=true",
    "agnews-mxbai": "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/agnews-mxbai-1024-euclidean.hdf5?download=true",
    "yahoo-minilm": "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/yahoo-minilm-384-normalized.hdf5?download=true",
    "gooaq-distilroberta": "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/gooaq-distilroberta-768-normalized.hdf5?download=true",
    "celeba-resnet": "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/celeba-resnet-2048-cosine.hdf5?download=true",
}


def _download(url: str, destination: Path):
    import certifi
    import ssl
    if not destination.is_file():
        logging.info(f"downloading {url} to {destination}")
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=context) as response:
            with open(destination, "wb") as out_file:
                out_file.write(response.read())


def compute_ground_truth(data, queries, k=100):
    def inner(row):
        distances = np.linalg.norm(data - row, axis=1)
        idxs = np.argsort(distances)
        idxs = idxs[:k]
        distances = distances[idxs]
        return distances, idxs

    zipped = Parallel(n_jobs=-2, prefer="threads")(
        delayed(inner)(row) for row in queries
    )
    distances, neighbors = zip(*zipped)
    return dict(distances=np.array(distances), neighbors=np.array(neighbors))
    

def preprocess(name: str, seed: int):
    logging.info(f"preprocessing {name}")
    path = DATA_DIR / Path(name + ".hdf5")
    path_public = DATA_DIR / Path(name + "-public" + ".hdf5")
    path_private = DATA_DIR / Path(name + "-private" + ".hdf5")
    _download(DATASET_URLS[name], path)
    with h5py.File(path) as hfp:
        orig_data = hfp["/train"][:]
        orig_queries = hfp["/test"][:]
    data = np.concat((orig_data, orig_queries))

    rng = np.random.default_rng(seed)
    rng.shuffle(data)

    queries_public = data[:1000]
    queries_private = data[1000:2000]
    dataset = data[2000:]

    logging.info("computing public ground truth")
    ground_public = compute_ground_truth(dataset, queries_public)
    logging.info("computing private ground truth")
    ground_private = compute_ground_truth(dataset, queries_private)

    for path, queries, ground in [
        (path_public, queries_public, ground_public),
        (path_private, queries_private, ground_private),
    ]:
        with h5py.File(path, "w") as hfp:
            hfp["/train"] = dataset
            hfp["/test"] = queries
            hfp["/neighbors"] = ground["neighbors"]
            hfp["/distances"] = ground["distances"]

    # Check that the data is the same but the queries are different
    assert np.all(h5py.File(path_private)["/train"][:] == h5py.File(path_public)["/train"][:])
    assert np.any(h5py.File(path_private)["/test"][:] != h5py.File(path_public)["/test"][:])


if __name__ == "__main__":
    seed = int(sys.argv[1])

    if not DATA_DIR.is_dir():
        DATA_DIR.mkdir()

    for dataset in DATASET_URLS.keys():
        preprocess(dataset, seed)
