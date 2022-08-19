# Globals
LOCATION="eastus"
DATESTAMP=$(date "+%Y%m%d")
RESOURCE_GROUP="azureml-assets-${DATESTAMP}"
WORKSPACE="azureml-assets-ws-${DATESTAMP}"
CPU_CLUSTER="cpu-cluster"
GPU_CLUSTER="gpu-cluster"

echo "Installing Azure CLI extension for Azure Machine Learning"
az extension add -n ml -y

echo "Configuring Azure CLI defaults"
az configure --defaults group=$RESOURCE_GROUP workspace=$WORKSPACE location=$LOCATION

# See if the last resource to be created already exists, and if so bail early
if az ml compute show --name $GPU_CLUSTER >/dev/null 2>&1; then
    echo "Azure resources already exist"
else
    resource_name="resource group ${RESOURCE_GROUP}"
    echo "Checking ${resource_name}"
    if ! az group show -n $RESOURCE_GROUP --output none >/dev/null 2>&1; then
        echo "Creating ${resource_name}"
        az group create -n $RESOURCE_GROUP -l $LOCATION
    fi
    
    resource_name="Machine Learning workspace ${WORKSPACE}"
    echo "Checking ${resource_name}"
    if ! az ml workspace show -n $WORKSPACE --output none >/dev/null 2>&1; then
        echo "Creating ${resource_name}"
        az ml workspace create -n $WORKSPACE
    fi

    resource_name="compute cluster ${CPU_CLUSTER}"
    echo "Checking ${resource_name}"
    if ! az ml compute show --name $CPU_CLUSTER --output none >/dev/null 2>&1; then
        echo "Creating ${resource_name}"
        az ml compute create --name $CPU_CLUSTER --size Standard_DS3_v2 --min-instances 0 --max-instances 10 --type AmlCompute
    fi

    resource_name="compute cluster ${GPU_CLUSTER}"
    echo "Checking ${resource_name}"
    if ! az ml compute show --name $GPU_CLUSTER --output none >/dev/null 2>&1; then
        echo "Creating ${resource_name}"
        az ml compute create --name $GPU_CLUSTER --size Standard_NC6 --min-instances 0 --max-instances 10 --type AmlCompute
    fi
fi

# Create environment variables
cat << EOF >> $GITHUB_ENV
resource_group=${RESOURCE_GROUP}
workspace=${WORKSPACE}
cpu_cluster=${CPU_CLUSTER}
gpu_cluster=${GPU_CLUSTER}
EOF
