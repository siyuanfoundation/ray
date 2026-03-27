#!/bin/bash
set -e
export LOCATION="us-east5"
export REPO_NAME="$USER-repo"
export PROJECT_ID=$(gcloud config get project)

gcloud auth configure-docker ${LOCATION}-docker.pkg.dev --quiet

export IMAGE_PATH=${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/ray-jax-data:v2.54.1

if [ -z "$GITHUB_USER" ]; then
    echo "Error: GITHUB_USER is not set."
    exit 1
fi

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN is not set."
    exit 1
fi

docker build --build-arg AUTH_TOKEN=$(gcloud auth print-access-token) --build-arg GITHUB_USER=${GITHUB_USER} --build-arg GITHUB_TOKEN=${GITHUB_TOKEN} -t ${IMAGE_PATH} .
docker push ${IMAGE_PATH}
echo "Build and push successful"
