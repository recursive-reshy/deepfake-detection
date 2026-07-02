# Base image
FROM python:3.10-slim

# System dependencies
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
