FROM ann-orthogonal/base:latest

# Install dependencies for compilation
RUN pip install --no-cache-dir pybind11

# Copy source files
COPY . /app/source

# Build and install the extension
WORKDIR /app/source
RUN pip install .

# Copy algorithm and scenarios to /app as required by the harness
WORKDIR /app
RUN cp /app/source/algorithm.py /app/algorithm.py && \
    cp /app/source/scenarios.yaml /app/scenarios.yaml

# No ENTRYPOINT override – the base image harness handles everything.
