FROM ann-orthogonal/base:latest

RUN pip install --no-cache-dir pybind11

# Copy self-contained source and build the hnsw_cpp extension.
# -march=native is intentional: the image is built on the machine that runs it.
COPY . /app/source
WORKDIR /app/source
RUN pip install .

# The harness loads /app/algorithm.py and extracts /app/scenarios.yaml.
WORKDIR /app
RUN cp /app/source/algorithm.py /app/algorithm.py && \
    cp /app/source/scenarios.yaml /app/scenarios.yaml

# No ENTRYPOINT override – the base image harness handles everything.
