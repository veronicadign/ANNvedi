# -----------------------------------------------------------------------
# NNS Competition – Student Dockerfile
#
# Build and push:
#   docker build -t <your-team>/nns:latest .
# -----------------------------------------------------------------------

FROM ann-orthogonal/base:latest

# ---- Install your algorithm's dependencies ----
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy your algorithm and scenario definitions ----
COPY algorithm.py  /app/algorithm.py
COPY scenarios.yaml /app/scenarios.yaml

# No ENTRYPOINT override – the base image harness handles everything.
