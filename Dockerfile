# syntax=docker/dockerfile:1
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
# Pinned, not :latest — a floating tag changes this layer's digest on every upstream uv
# release, which busts every layer below it (including the uv sync layer two steps down)
# even when pyproject.toml/uv.lock haven't changed. That silently forced a full
# TensorFlow re-download on unrelated builds; this pin is what actually fixes it, not
# just the layer ordering below (which was already correct).
COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /usr/local/bin/uv

# Working directory
WORKDIR /app
ENV PYTHONPATH=/app

# Install dependencies
# Copy lockfile and project metadata first — layer caches until these change.
# --mount=type=cache persists uv's package cache across builds in a BuildKit cache mount
# (not a Docker layer, so it survives even when this RUN layer's cache correctly
# invalidates on a genuine uv.lock change) — only the actual delta re-downloads rather
# than all 545MB of TensorFlow again.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
	uv sync --frozen --no-dev

# Copy application code
COPY . .

# Port
EXPOSE 8080

# Default entrypoint — Cloud Run (Uvicorn)
# Vertex AI overrides this at job submission time with:
#   python src/training/train.py --job-id {job_id}
CMD [ "uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080" ]
