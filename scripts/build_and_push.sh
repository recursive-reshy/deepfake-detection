#!/usr/bin/env bash
# =============================================================================
# build_and_push.sh
# Single source of truth for IMAGE_URI — builds, pushes, resolves the pushed image's
# content digest, and writes it into .env as IMAGE_URI=...@sha256:<digest>. Every
# consumer of IMAGE_URI (api/routes/train.py's Stage 1 submission, vertex_job.yaml's
# Stage 2 self-submission, a manual `gcloud run deploy`) reads from this one place —
# no second script or command types or infers the digest independently.
#
# Root cause this fixes: tagging by git SHA alone (:${git_sha}) is still a mutable
# reference — rebuilding/repushing the same commit (a routine dev-loop occurrence, not
# just a rebase/force-push edge case) silently moves what that tag points at. Vertex AI
# or Cloud Run can then resolve the tag to a stale cached digest at execution time while
# the source looks correct — the root cause of a prior multi-cycle debugging detour.
# Pinning to the digest this script resolves, never the tag, removes that ambiguity.
#
# Requires AR_IMAGE_BASE in .env (written by scripts/setup_gcp.sh).
# Usage: scripts/build_and_push.sh
# =============================================================================

set -euo pipefail

source .env

GIT_SHA=$(git rev-parse --short HEAD)
IMAGE_TAG="${AR_IMAGE_BASE}:${GIT_SHA}"

echo "Building ${IMAGE_TAG} (GIT_SHA=${GIT_SHA})..."
docker build --build-arg GIT_SHA="${GIT_SHA}" -t "${IMAGE_TAG}" .

echo "Pushing ${IMAGE_TAG}..."
docker push "${IMAGE_TAG}"

DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "${IMAGE_TAG}")

if [[ -z "${DIGEST}" ]]; then
  echo "ERROR: could not resolve a digest for ${IMAGE_TAG} — refusing to write an empty IMAGE_URI" >&2
  exit 1
fi

# Single write path for IMAGE_URI — replace the existing line in .env, or append if this
# is the first build since .env was created.
if grep -q '^IMAGE_URI=' .env; then
  sed -i.bak "s|^IMAGE_URI=.*|IMAGE_URI=${DIGEST}|" .env
  rm -f .env.bak
else
  echo "IMAGE_URI=${DIGEST}" >> .env
fi

echo "IMAGE_URI resolved and written to .env:"
echo "  ${DIGEST}"
