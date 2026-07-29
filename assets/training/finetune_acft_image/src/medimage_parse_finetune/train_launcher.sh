#!/bin/bash

echo "=== CxrReportGen Train Launcher Debug ==="
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE" 
echo "LOCAL_RANK: $LOCAL_RANK"
echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "============================================"

# Run the main application
echo "Starting training with args: $@"
python -m azureml.acft.image.components.olympus.app.main "$@"