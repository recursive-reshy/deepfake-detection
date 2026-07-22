#!/usr/bin/env bash
# =============================================================================
# build_and_push.sh
# Builds and pushes the Docker image, then prints the pushed image's content digest.
# Digest, not tag, is what every consumer of IMAGE_URI must be pointed at (Cloud Run
# deploy, api/routes/train.py's Stage 1 submission, vertex_job.yaml's Stage 2
# self-submission via config.IMAGE_URI). This script does NOT write .env itself —
# it only resolves the digest and prints it; copying that value into .env (or into
# Cloud Run's --set-env-vars) is a deliberate, visible step you take yourself, not
# something a build script should do silently on your behalf.
#
# Root cause this responds to: tagging by git SHA alone (:${git_sha}) is still a
# mutable reference — rebuilding/repushing the same commit (a routine dev-loop
# occurrence, not just a rebase/force-push edge case) silently moves what that tag
# points at. Vertex AI or Cloud Run can then resolve the tag to a stale cached digest
# at execution time while the source looks correct — the root cause of a prior
# multi-cycle debugging detour. Pinning to the digest this script resolves, never the
# tag, removes that ambiguity.
#
# Requires GCP_PROJECT_ID and GCP_REGION in .env (both already required elsewhere —
# see config.py).
#
# Usage: scripts/build_and_push.sh
# =============================================================================

set -euo pipefail

source .env

AR_REPO="deepfake-detection"
IMAGE_BASE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/deepfake-detection"

GIT_SHA=$(git rev-parse --short HEAD)
IMAGE_TAG="${IMAGE_BASE}:${GIT_SHA}"

echo "Building ${IMAGE_TAG} (GIT_SHA=${GIT_SHA})..."
docker build --build-arg GIT_SHA="${GIT_SHA}" -t "${IMAGE_TAG}" .

echo "Pushing ${IMAGE_TAG}..."
docker push "${IMAGE_TAG}"

DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "${IMAGE_TAG}")

if [[ -z "${DIGEST}" ]]; then
  echo "ERROR: could not resolve a digest for ${IMAGE_TAG}" >&2
  exit 1
fi

echo ""
echo "Pushed image digest resolved:"
echo "  IMAGE_URI=${DIGEST}"
echo ""
echo "Copy this into .env's IMAGE_URI before submitting a job, and into Cloud Run's"
echo "--image / --set-env-vars IMAGE_URI=... before deploying."
