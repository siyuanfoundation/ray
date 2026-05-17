#!/bin/bash
set -e
set -x
export LOCATION="us-east5"
export REPO_NAME="$USER-repo"
PROJECT_ID=$(gcloud config get project)
export PROJECT_ID

gcloud auth configure-docker "${LOCATION}-docker.pkg.dev" --quiet

BUILD_DATE="$(date +%Y%m%d)"
BUILD_DATE="v7"
if [ -z "$GITHUB_USER" ]; then
    echo "Error: GITHUB_USER is not set."
    exit 1
fi

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN is not set."
    exit 1
fi

export IMAGE_PATH="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/ray-torch-tpu:${BUILD_DATE}"
docker image rm "${IMAGE_PATH}" || true
docker build --build-arg BUILD_DATE="$BUILD_DATE" --build-arg AUTH_TOKEN="$(gcloud auth print-access-token)" -t "${IMAGE_PATH}" -f torch_tpu/train.Dockerfile .
docker push "${IMAGE_PATH}"
echo "Build and push successful: ${IMAGE_PATH}"

# export IMAGE_PATH="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/ray-torch-tpu-vllm:${BUILD_DATE}"
# docker image rm "${IMAGE_PATH}"
# docker build --build-arg BUILD_DATE="$BUILD_DATE" --build-arg AUTH_TOKEN="$(gcloud auth print-access-token)" --build-arg GITHUB_USER="${GITHUB_USER}" --build-arg GITHUB_TOKEN="${GITHUB_TOKEN}" -t "${IMAGE_PATH}" -f vllm.Dockerfile ..
# docker push "${IMAGE_PATH}"
# echo "Build and push successful: ${IMAGE_PATH}"
