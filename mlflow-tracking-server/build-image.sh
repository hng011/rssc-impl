#!/bin/bash

set -e

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

docker build -t ${LOCAL_REPO}/${IMAGE_NAME}:$VERSION .

# TAG LOCAL
docker tag ${LOCAL_REPO}/${IMAGE_NAME}:$VERSION ${LOCAL_REPO}/${IMAGE_NAME}:latest

# TAG FOR ACR
docker tag ${LOCAL_REPO}/${IMAGE_NAME}:$VERSION ${ACR_REPO}/${IMAGE_NAME}:$VERSION
docker tag ${LOCAL_REPO}/${IMAGE_NAME}:$VERSION ${ACR_REPO}/${IMAGE_NAME}:latest

echo "DONE BUILDING :)"