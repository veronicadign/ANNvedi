FROM ann-orthogonal/base:latest

RUN pip install --no-cache-dir pybind11

COPY . /app/source
WORKDIR /app/source
RUN pip install .

WORKDIR /app
RUN cp /app/source/algorithm.py  /app/algorithm.py && \
    cp /app/source/scenarios.yaml /app/scenarios.yaml

# No ENTRYPOINT override – the base image harness handles everything.
