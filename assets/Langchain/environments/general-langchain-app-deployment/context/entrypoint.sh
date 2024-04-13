#!/bin/bash

set -x

cd $AZUREML_MODEL_DIR

dir_count=$(find "$AZUREML_MODEL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

if [ "$dir_count" -eq 1 ]; then
    app_folder=$(find "$AZUREML_MODEL_DIR" -mindepth 1 -maxdepth 1 -type d)
    cd "$app_folder"
    echo "Changed directory to $(pwd)"
fi
ls

poetry install --no-interaction --no-ansi

exec uvicorn app.server:app --host 0.0.0.0 --port 8080
