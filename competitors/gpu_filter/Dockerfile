FROM ann-orthogonal/base:latest

# nvcc + headers from the NVIDIA apt repo. The image is BUILT without a GPU:
# the kernel is compiled to compute_80 PTX and JIT-compiled by the driver on
# whatever GPU the container sees at run time (incl. Blackwell). cudart is
# linked statically, so no CUDA runtime libraries are needed in the image —
# the driver stack is injected by the NVIDIA container runtime (--gpus).
# the base image enables only 'main'; nvcc lives in contrib/non-free
RUN sed -i 's/Components: main/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/source
WORKDIR /app/source

RUN nvcc -O3 -shared -Xcompiler -fPIC \
        -gencode arch=compute_80,code=compute_80 \
        src/gpu_filter.cu -o /app/libgpufilter.so

WORKDIR /app
RUN cp /app/source/algorithm.py /app/algorithm.py && \
    cp /app/source/scenarios.yaml /app/scenarios.yaml

# No ENTRYPOINT override – the base image harness handles everything.
# NOTE: the evaluator must run this container with --gpus (NVIDIA container
# toolkit); fit() fails with a clear error otherwise.
