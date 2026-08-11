FROM ann-orthogonal/base:latest

# sklearn/scipy no longer needed: fit-time GP tuning was replaced by offline
# per-dataset tuning (scripts/tune_parameters.py -> scenarios.yaml)
RUN pip install --no-cache-dir pybind11

# Copy self-contained source files
COPY . /app/source

# Build and install the extension
WORKDIR /app/source
RUN pip install .

# Copy algorithm and scenarios to /app as required by the harness
WORKDIR /app
RUN cp /app/source/algorithm.py /app/algorithm.py && \
    cp /app/source/scenarios.yaml /app/scenarios.yaml

# No ENTRYPOINT override – the base image harness handles everything.
