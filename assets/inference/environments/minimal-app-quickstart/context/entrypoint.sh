#!/bin/bash

set -x

# Set STARTUP_SCRIPT to a default value if it is not set or is empty
STARTUP_SCRIPT=${STARTUP_SCRIPT:-startup.sh}

export

dir_count=$(find "$AZUREML_MODEL_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

if [ "$dir_count" -eq 1 ]; then
    # Get the name of the directory
    app_folder=$(find "$AZUREML_MODEL_DIR" -mindepth 1 -maxdepth 1 -type d)
    # Change to that directory
    cd "$app_folder"
    echo "Changed directory to $(pwd)"
    ls
    bash $STARTUP_SCRIPT
else
    echo "Directory error found in $AZUREML_MODEL_DIR, please check the directory structure and make sure there is only one folder for the app."
    exit 1
fi

