#!/bin/sh

set -e

echo "Starting MLflow Tracking Server..."
echo "Backend Store URI: $BACKEND_STORE_URL"
echo "Artifact Root: $ARTIFACT_ROOT"
echo "Port: $MLFLOW_PORT"

mlflow server \
  --backend-store-uri "$BACKEND_STORE_URL" \
  --default-artifact-root "$ARTIFACT_ROOT" \
  --host 0.0.0.0 \
  --port "$MLFLOW_PORT"