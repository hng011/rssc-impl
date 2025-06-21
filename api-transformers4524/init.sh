#!/bin/bash
set -e  

cd ./app

export AZURE_STORAGE_ACCESS_KEY="$AZURE_STORAGE_ACCESS_KEY"
echo "AZURE_STORAGE_ACCESS_KEY: ${AZURE_STORAGE_ACCESS_KEY:0:3}"

export APP_PORT="$APP_PORT"
echo "APP PORT: ${APP_PORT}"

uvicorn main:app --host 0.0.0.0 --port ${APP_PORT}