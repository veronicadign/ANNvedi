FROM ann-orthogonal/base:latest

# nvcc from Debian non-free (the base enables only 'main') + pybind11 for hnsw
RUN sed -i 's/Components: main/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pybind11

COPY . /app/source
WORKDIR /app/source

# CPU backend: hnsw_cpp via pip; GPU backend: setup.py's build_gpu command
# (nvcc -> libgpufilter.so, compute_80 PTX so GPU-less builds JIT on target)
RUN pip install .
RUN python3 setup.py build_gpu && cp libgpufilter.so /app/libgpufilter.so

WORKDIR /app
RUN cp /app/source/algorithm.py /app/algorithm.py && \
    cp /app/source/scenarios.yaml /app/scenarios.yaml

# No ENTRYPOINT override – the base image harness handles everything.
# With --gpus the gpu blocks run on the GPU; without it every block
# falls back to the tuned hnsw parameters it carries.
