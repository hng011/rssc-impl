#!/bin/bash

source .env

# Parse optional overrides: -e VAR VALUE
while getopts "e:" opt; do
  case $opt in
    e)
      VAR_NAME=${OPTARG}
      shift $((OPTIND - 1))
      VALUE=$1
      if [ -n "$VAR_NAME" ] && [ -n "$VALUE" ]; then
        export "$VAR_NAME"="$VALUE"
        echo "Setting $VAR_NAME TO $VALUE"
        shift
      else
        echo "Usage: -e VAR_NAME VALUE"
        exit 1
      fi
      ;;
    *)
      echo "Invalid option"
      exit 1
      ;;
  esac
done

echo "running ${MLFLOW_DOCKER_IMAGE}:${MLFLOW_DOCKER_IMAGE_TAG}"

docker run --env-file .env \
    -p 8080:${MLFLOW_PORT} \
    -e MLFLOW_PORT=${MLFLOW_PORT} \
    -e BACKEND_STORE_URI="${BACKEND_STORE_URI}" \
    -e ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
    ${MLFLOW_DOCKER_IMAGE}:${MLFLOW_DOCKER_IMAGE_TAG}