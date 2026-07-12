# Base image — TensorFlow's official GPU image, not python:3.10-slim + tensorflow[and-cuda].
# python:3.10-slim never engaged the GPU on Vertex AI: it lacks the NVIDIA Container
# Toolkit-compatible mount points (LD_LIBRARY_PATH=/usr/local/nvidia/lib{,64},
# NVIDIA_VISIBLE_DEVICES, NVIDIA_DRIVER_CAPABILITIES, /etc/ld.so.conf.d/*cuda*.conf) that
# Vertex AI's GPU runtime relies on to mount the host driver's libcuda.so into the
# container — pip-installed CUDA/cuDNN toolkit libraries (tensorflow[and-cuda]'s
# nvidia-cu12-* wheels) never included libcuda.so itself (it must match the host driver
# exactly, so no image — official or not — bakes it in) and can't substitute for this
# missing OS-level plumbing. This base image ships that plumbing already configured and
# guaranteed to match its own bundled CUDA 12/cuDNN 9 build. See the two prior handoff
# briefs on this fix for the full cuInit/libcuda.so failure history.
FROM tensorflow/tensorflow:2.21.0-gpu

# System dependencies
# curl is already present in this base image (Ubuntu 22.04); kept explicit here rather
# than assumed, since the image itself is not guaranteed to be the same across future
# TensorFlow version bumps to this FROM line.
RUN apt-get update && apt-get install -y --no-install-recommends \
	curl \
	&& rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Working directory
WORKDIR /app
ENV PYTHONPATH=/app

# Install dependencies
# Copy lockfile and project metadata first — layer caches until these change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Port
EXPOSE 8080

# Default entrypoint — Cloud Run (Uvicorn)
# Vertex AI overrides this at job submission time with:
#   python src/training/train.py --job-id {job_id}
CMD [ "uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080" ]
