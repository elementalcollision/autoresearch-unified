FROM runpod/pytorch:2.6.0-py3.11-cuda12.8.1-devel-ubuntu22.04

WORKDIR /workspace/autoresearch-unified

# Python deps (torch is pre-installed in RunPod base image)
RUN pip install --no-cache-dir \
    numpy>=2.2.6 \
    pyarrow>=21.0.0 \
    requests>=2.32.0 \
    rustbpe>=0.1.0 \
    tiktoken>=0.11.0 \
    textual>=3.0.0 \
    anthropic>=0.40.0 \
    matplotlib>=3.10.8 \
    pandas>=2.3.3 \
    ipywidgets

# Copy project source
COPY . .

# Place notebooks in JupyterLab's default file browser location
RUN cp -r platforms/runpod/notebooks /workspace/notebooks

# Point data cache at network volume for persistence across pod restarts
ENV AUTORESEARCH_CACHE_DIR=/workspace/.cache/autoresearch
ENV AUTORESEARCH_BACKEND=cuda
ENV PYTHONPATH=/workspace/autoresearch-unified
