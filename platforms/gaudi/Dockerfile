FROM vault.habana.ai/gaudi-docker/1.23.0/ubuntu24.04/habanalabs/pytorch-installer-2.5.1:latest

WORKDIR /workspace/autoresearch

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (torch is pre-installed in the Habana base image — do NOT pip install torch)
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    numpy>=2.2.6 \
    pyarrow>=21.0.0 \
    requests>=2.32.0 \
    rustbpe>=0.1.0 \
    tiktoken>=0.11.0

# Optional: TUI + agent dependencies
RUN pip install --no-cache-dir \
    textual>=3.0.0 \
    anthropic>=0.40.0

# Source habanalabs environment on shell login
RUN echo "source /etc/profile.d/habanalabs*.sh 2>/dev/null || true" >> /root/.bashrc

# Copy source code
COPY . .

# Default: run single-HPU training
CMD ["python", "-u", "train_gaudi.py"]
