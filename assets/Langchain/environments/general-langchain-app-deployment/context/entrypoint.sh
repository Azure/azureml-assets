#!/bin/bash

# Print commands and their arguments as they are executed
set -x

cd $AZUREML_MODEL_DIR

# Count the number of directories within the specified directory
dir_count=$(find "$AZUREML_MODEL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

if [ "$dir_count" -eq 1 ]; then
    # Get the name of the directory
    app_folder=$(find "$AZUREML_MODEL_DIR" -mindepth 1 -maxdepth 1 -type d)
    # Change to that directory
    cd "$app_folder"
    echo "Changed directory to $(pwd)"
fi
ls

# Install any dependencies (if not already installed)
poetry install --no-interaction --no-ansi

# Start the Uvicorn server
exec uvicorn app.server:app --host 0.0.0.0 --port 8080
