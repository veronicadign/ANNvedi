Orthogonal school competition
=============================

This is the evaluation harness for the second year's competition of the
[Orthogonal school](https://www.elicsir.it/en/orthogonal-school), focusing
on k-nearest neighbors search.


Submission format
-----------------

A submission consists of a docker image that extends the
[base image](https://github.com/Cecca/orthogonal-competition/blob/main/Dockerfile)
of the competition. Such image should should contain two files:

- `/app/algorithm.py`: this is a python module defining an `Algorithm` class with
  three methods:
    - `fit` to train the data structure on the dataset to be queried
    - `query` to answer a single k-nn query
    - `get_n_distances` to return the number of distances computed since the instantiation
      of the algorithm
- `/app/scenarios.yaml`: a configuration file that specifies the parameters of the algorithm
  for each execution scenario and, possibly, dataset.

The `template` directory gives a customizable template for the submissions. In particular,
[`algorithm.py`](https://github.com/Cecca/orthogonal-competition/blob/main/template/algorithm.py)
and
[`scenarios.yaml`](https://github.com/Cecca/orthogonal-competition/blob/main/template/scenarios.yaml)
are of direct interest.
The 
[`Dockerfile`](https://github.com/Cecca/orthogonal-competition/blob/main/template/Dockerfile)
gives an example to build the `Dockerfile`.

The `competitors/faiss-hnsw` shows how to integrate the popular `faiss-hnsw` baseline
in this competition.

Running the competition
-----------------------

Assuming datasets are in the `datasets` directory, for a team `team` with a docker image
`team-image` we can the benchmark with the following command:

```
python evaluator.py evaluate --team team --image team-image \
                             --dataset data/data-file.hdf5
```

The evaluator:
  1. Extracts scenarios.yaml from the image (without running it) to discover
     scenario names.
  2. Spawns one fresh container per scenario, passing SCENARIO_NAME so the
     harness runs exactly that configuration.
  3. While the container runs, polls Docker's memory stats in a background
     thread to record the true peak RSS (cgroup-based, includes all C heap).
  4. Reads the flat results.hdf5 written by the harness and computes metrics.
  5. Stores one DB row per scenario.


Championships
--------

Submissions are ranked in the following championships:

- _Sherlock Holmes_: the fastest approach achieving average recall above 0.95 wins
- _Bianconiglio_: the fastest approach achieving average recall above 0.8 wins
- _Dory_: the approach using the least amount of memory while achieving recall above 0.95,
  while at the same time not taking more than twice the time of the
  [`faiss-hnsw`](https://github.com/Cecca/orthogonal-competition/tree/main/competitors/faiss-hnsw)
  baseline with parameters `efConstruction: 100, M: 16, ef: 50`.
- _Marie Kondo_: the fastest approach at building the index, in a configuration that achieves at least 0.95 recall
- _Paperone_: the approache making the fewest distance computations, with an average recall at least 0.95.
  For the purpose of this championship, only _full Euclidean distance computations_ count.
  That is, sketches and product quantization (and similar techniques) _do not_ count towards the distance count.

  
Scenarios
---------

The submissions will be run in different scenarios:

- `high_recall`: submissions should aim for a recall of at least 0.95. This scenario
  is used for the _Sherlock Holmes_, _Marie Kondo_, and _Paperone_ prizes;
- `fast`: submissions should aim for a recall of at least 0.8. This scenario determines
  the _Bianconiglio_ prize;
- `memory`: submissions should aim for a recall of at least 0.95. This scenario
  is used for the _Dory_ prize.




