location="eastus"
datestamp=$(date "+%Y%m%d")
resource_group="azureml-assets-${datestamp}"
container_registry="azuremlassetscr${datestamp}"
workspace="azureml-assets-ws-${datestamp}"
cpu_cluster="cpu-cluster"
gpu_cluster="gpu-cluster"
gpu_v100_cluster="gpu-v100-cluster"

echo "Installing Azure CLI extension for Azure Machine Learning"
az extension add -n ml -y

echo "Configuring Azure CLI defaults"
az configure --defaults group=$resource_group workspace=$workspace location=$location

# See if the last resource to be created already exists, and if so bail early
if az ml compute show --name $gpu_v100_cluster >/dev/null 2>&1; then
    echo "Azure resources already exist"
else
    resource_name="resource group ${resource_group}"
    echo "Checking ${resource_name}"
    if ! az group show -n $resource_group --output none >/dev/null 2>&1; then
        echo "Creating ${resource_name}"
        az group create -n $resource_group -l $location
    fi
    
    resource_name="container registry ${container_registry}"
    echo "Checking ${resource_name}"
    if ! az acr show -n $container_registry --output none >/dev/null 2>&1; then
        echo "Creating ${resource_name}"
        az acr create -n $container_registry --sku Basic --admin-enabled
    fi
    container_registry_id=$(az acr show -n $container_registry --query "id" -o tsv | tr -d '\n\r')
    
    resource_name="Machine Learning workspace ${workspace}"
    echo "Checking ${resource_name}"
    if ! az ml workspace show -n $workspace --output none >/dev/null 2>&1; then
        echo "Creating ${resource_name}"
        az ml workspace create -n $workspace
    fi

    resource_name="compute cluster ${cpu_cluster}"
    echo "Checking ${resource_name}"
    if ! az ml compute show --name $cpu_cluster --output none >/dev/null 2>&1; then
        echo "Creating ${resource_name}"
        az ml compute create --name $cpu_cluster --size Standard_DS3_v2 --min-instances 0 --max-instances 10 --type AmlCompute --idle-time-before-scale-down 120
    fi

    resource_name="compute cluster ${gpu_cluster}"
    echo "Checking ${resource_name}"
    if ! az ml compute show --name $gpu_cluster --output none >/dev/null 2>&1; then
        echo "Creating ${resource_name}"
        az ml compute create --name $gpu_cluster --size Standard_NC4as_T4_v3 --min-instances 0 --max-instances 10 --type AmlCompute --idle-time-before-scale-down 120
    fi

    resource_name="compute cluster ${gpu_v100_cluster}"
    echo "Checking ${resource_name}"
    if ! az ml compute show --name $gpu_v100_cluster --output none >/dev/null 2>&1; then
        echo "Creating ${resource_name}"
        az ml compute create --name $gpu_v100_cluster --size Standard_NC4as_T4_v3 --min-instances 0 --max-instances 2 --type AmlCompute --idle-time-before-scale-down 120
    fi
fi

# Create environment variables
cat << EOF >> $GITHUB_ENV
resource_group=${resource_group}
container_registry=${container_registry}
workspace=${workspace}
cpu_cluster=${cpu_cluster}
gpu_cluster=${gpu_cluster}
gpu_v100_cluster=${gpu_v100_cluster}
EOF
