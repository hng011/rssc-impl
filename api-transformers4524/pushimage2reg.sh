#!/bin/bash

source .env

VERSION=""

while getopts "t:" opt; do
  case $opt in
    t)
      VERSION="$OPTARG"
      echo "Version set to: $VERSION"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

az acr login --name ${AZURE_REGISTERY_NAME}

echo "Pushing ${LOCAL_REPO}/${IMAGE_NAME}:$VERSION to ${ACR_REPO}/${IMAGE_NAME}:$VERSION"
docker push ${ACR_REPO}/${IMAGE_NAME}:$VERSION

echo "Pushing ${LOCAL_REPO}/${IMAGE_NAME}:latest to ${ACR_REPO}/${IMAGE_NAME}:latest"
docker push ${ACR_REPO}/${IMAGE_NAME}:latest