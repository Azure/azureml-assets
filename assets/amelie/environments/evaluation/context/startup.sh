#!/bin/bash

# DATA_MOUNT_PARAMS="<storage_account_name>:<container_name>[:<sas_token>]"
# WS_MOUNT_PARAMS="<storage_account_name>:<container_name>[:<sas_token>]"
# SYMLINK_PAIRS="/mnt/data:/workspace/data;/mnt/config:/workspace/config;"
# WORKSPACE_DIR="/workspace"
# DEBUG_MODE="1" #> For showing startup.sh outputs
# COPY_OUTPUTS="1" #> For copying outputs to /ws/ directory

if [[ "${DEBUG_MODE}" != "1" ]]; then
  exec 3>&1
  exec 1>/dev/null
fi

echo "=== AZUREML Environment Variables ==="
env | grep -i "^azureml" | sort

pushd /workspace

MOUNT_PATH="${MOUNT_PATH:-/workspace}"

# Global counter for blob storage mounts
BLOB_MOUNT_COUNTER=1

# Function to mount blob storage
mount_blob_storage() {
  local storage_account_name="$1"
  local container_name="$2"
  local sas_token="$3"
  
  # Generate mount path using storage account name
  local mount_path="${MOUNT_PATH}/${storage_account_name}/${container_name}"
  
  # Generate unique paths using counter
  local config_file_path="/workspace/blob-config-${BLOB_MOUNT_COUNTER}.yml"
  local block_cache_path="/tmp/blob_cache/${BLOB_MOUNT_COUNTER}"
  local file_cache_path="/tmp/file_cache/${BLOB_MOUNT_COUNTER}"
  
  echo "Mounting blob storage #${BLOB_MOUNT_COUNTER} to: $mount_path"
  
  # Create necessary directories
  mkdir -p "$block_cache_path"
  mkdir -p "$file_cache_path"
  mkdir -p "$mount_path"
  
  # Create blob configuration file
  cat > "$config_file_path" << EOF
logging:
  level: log_warning
block_cache:
  path: $block_cache_path
file_cache:
  path: $file_cache_path
azstorage:
  account-name: $storage_account_name
  container: $container_name
EOF

  # Append authentication-specific config
  if [[ -n "$sas_token" ]]; then
    cat >> "$config_file_path" << EOF
  mode: sas
  sas: $sas_token
EOF
  else
    cat >> "$config_file_path" << EOF
  mode: msi
EOF
  fi

  # Mount the blob storage
  if blobfuse2 mount "$mount_path" --config-file="$config_file_path"; then   
    # Increment counter for next mount
    ((BLOB_MOUNT_COUNTER++))
    return 0
  else
    echo "Error: Failed to mount blob storage #${BLOB_MOUNT_COUNTER}" >&2
    return 1
  fi
}

# Function to parse and mount blob storage from parameter string
parse_and_mount_blob_storage() {
  local mount_params="$1"
  local mount_name="$2"
  
  if [[ -z "$mount_params" ]]; then
    echo "No $mount_name mount parameters provided" >&2
    return 0
  fi
  
  # Parse the parameter string: <storage_account_name>:<container_name>:<sas_token>
  IFS=':' read -r storage_account_name container_name sas_token <<< "$mount_params"
  
  # Validate required fields
  if [[ -z "$storage_account_name" || -z "$container_name" ]]; then
    echo "Error: Invalid $mount_name format. Expected: <storage_account_name>:<container_name>[:<sas_token>]"
    echo "Received: $mount_params"
    return 1
  fi
  
  # Mount the blob storage
  mount_blob_storage "$storage_account_name" "$container_name" "$sas_token"
}

# Mount data storage if DATA_MOUNT_PARAMS is provided
if [[ -n "$DATA_MOUNT_PARAMS" ]]; then
  parse_and_mount_blob_storage "$DATA_MOUNT_PARAMS" "DATA_MOUNT_PARAMS"
fi

# Mount workspace storage if WS_MOUNT_PARAMS is provided
if [[ -n "$WS_MOUNT_PARAMS" ]]; then
  parse_and_mount_blob_storage "$WS_MOUNT_PARAMS" "WS_MOUNT_PARAMS"
fi

# Create symlinks
if [[ -n "$SYMLINK_PAIRS" ]]; then
  IFS=';' read -ra PAIR_ARRAY <<< "$SYMLINK_PAIRS"
  for pair in "${PAIR_ARRAY[@]}"; do
    IFS=':' read -r source target <<< "$pair"

    if [[ -z "$source" || -z "$target" ]]; then
      echo "Skipping invalid pair: $pair"
      continue
    fi

    source="${source//\$AZUREML_CR_DATA_CAPABILITY_PATH/$AZUREML_CR_DATA_CAPABILITY_PATH}"
    
    if [[ ! -e "$source" ]]; then
      echo "Creating source directory $source"
      mkdir -p "$source"
    fi

    echo "Creating symlink: $target -> $source"
    echo "Creating target directory $target"
    mkdir -p "$(dirname "$target")" 
    ln -sf "$source" "$target"
  done
fi

popd

if [[ -n "$WORKSPACE_DIR" ]]; then
  echo "Setting workspace directory to: $WORKSPACE_DIR"
  cd "$WORKSPACE_DIR" || exit 1
else
  echo "No workspace directory set. Using default directory"
fi

MARKER_FILE="/tmp/start_$(date +%s%N)"
touch "$MARKER_FILE"

# Determine output directory
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
if [[ -n "$AZUREML_CR_DATA_CAPABILITY_PATH" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR//\$AZUREML_CR_DATA_CAPABILITY_PATH/$AZUREML_CR_DATA_CAPABILITY_PATH}"
fi

echo "Ouput directory: $OUTPUT_DIR"

if [[ "${DEBUG_MODE}" != "1" ]]; then
  exec 1>&3
  exec 3>&-
fi

terminate() {
  kill -TERM "$child_pid" 2>/dev/null
}
trap terminate TERM INT

"$@" &
child_pid=$!

wait "$child_pid"
EXIT_CODE=$?

COPY_OUTPUTS="${COPY_OUTPUTS:-1}"
if [[ "$COPY_OUTPUTS" == "1" ]]; then
  rsync -av --delete \
    --exclude="__pycache__" \
    --exclude="azureml-logs" \
    --exclude="logs" \
    --exclude="user_logs" ./ /ws/ || true
fi

mkdir -p "$OUTPUT_DIR"
find . -name outputs -prune -o \
       -name artifacts -prune -o \
       -name user_logs -prune -o \
       -name __pycache__ -prune -o \
       -type f -newer "$MARKER_FILE" -print0 | while IFS= read -r -d '' file; do
  cp --parents "$file" "$OUTPUT_DIR"/
  echo "$file|$(stat -c%s "$file")" >> "$OUTPUT_DIR/.generated"
done

exit $EXIT_CODE