#!/usr/bin/env bash
# =============================================================================
# dev_run.sh
# Local iteration path for src/ and api/ changes — bypasses Docker's build and
# Vertex AI's job-provisioning wait entirely for pure code changes (no dependency
# changes). Bind-mounts the current src/ and api/ directories into an already-built
# image at container start, so edits are picked up on the next run with no rebuild
# and no `uv sync` step.
#
# This is what would have caught the --stage argparse gap and the YAML args-format
# issue in seconds, on this machine, with zero GPU/build/Vertex AI cost.
#
# Usage:
#   docker build -t deepfake-detection:dev .        # once, or whenever deps change
#   scripts/dev_run.sh python src/training/train.py --job-id test --stage base
#   scripts/dev_run.sh python scripts/validate_local.py
#
# DEV_IMAGE overrides the image tag (default: deepfake-detection:dev).
# =============================================================================

set -euo pipefail

DEV_IMAGE="${DEV_IMAGE:-deepfake-detection:dev}"

docker run --rm \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/api:/app/api" \
  -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
  -v "$(pwd)/service-account-key.json:/app/service-account-key.json:ro" \
  --env-file .env \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/service-account-key.json \
  "${DEV_IMAGE}" \
  uv run --frozen --no-dev "$@"