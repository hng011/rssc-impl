#!/bin/bash

source .env

echo "running ${LOCAL_REPO}/${IMAGE_NAME}:latest"

docker run --env-file .env \
    -p ${HOST_PORT}:${MLFLOW_PORT} \
    -e MLFLOW_PORT=${MLFLOW_PORT} \
    -e BACKEND_STORE_URI="${BACKEND_STORE_URI}" \
    -e ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
    -e AZURE_STORAGE_ACCESS_KEY="${AZURE_STORAGE_ACCESS_KEY}" \
    ${LOCAL_REPO}/${IMAGE_NAME}:latest