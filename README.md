# ANNvedi

Currently (BASELINE): linear (brute-force) k-nearest neighbor index = for each query we compute L2 distance to all dataset vectors and select k smallest

with a complexity of O(N × D) where N is the number of vectors and D is the vector dimension

install the requirements for Python environment:
```
pip install -r requirements.txt
```
build the C++ module:
```
pip install -e .
```
