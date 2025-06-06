#!/bin/bash

source .env

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

az acr login --name ${AZURE_REGISTERY_NAME}

echo "Pushing ${MLFLOW_DOCKER_IMAGE}:${MLFLOW_DOCKER_IMAGE_TAG} to Azure ${AZURE_REGISTERY_NAME}.azurecr.io"

docker push ${MLFLOW_DOCKER_IMAGE}:${MLFLOW_DOCKER_IMAGE_TAG}