#!/bin/bash

set -e

echo "Starting MLflow Tracking Server..."
echo "Backend Store URI: $BACKEND_STORE_URL"
echo "Artifact Root: $ARTIFACT_ROOT"
echo "Port: $MLFLOW_PORT"

export AZURE_STORAGE_ACCESS_KEY="$AZURE_STORAGE_ACCESS_KEY"
echo "AZURE_STORAGE_ACCESS_KEY: ${AZURE_STORAGE_ACCESS_KEY:0:3}"

mlflow server \
  --backend-store-uri "$BACKEND_STORE_URI" \
  --default-artifact-root "$ARTIFACT_ROOT" \
  --host 0.0.0.0 \
  --port "$MLFLOW_PORT"