# Globals
LOCATION="eastus"
RESOURCE_GROUP_PREFIX="azureml-assets"
DATESTAMP=$(date "+%Y-%m-%d")
RESOURCE_GROUP_NAME="${RESOURCE_GROUP_PREFIX}-${DATESTAMP}"

echo "Retrieving subscription information..."
SUBSCRIPTION_NAME=$(az account show --query name -o tsv | tr -d '\n\r')
SUBSCRIPTION_ID=$(az account show --query id -o tsv | tr -d '\n\r')

echo "Installing Azure CLI extension for Azure Machine Learning..."
az extension add -n ml -y

echo "Creating resource group ${RESOURCE_GROUP_NAME}..."
az group create -n $RESOURCE_GROUP_NAME -l $LOCATION

exit

echo "Creating Azure Machine Learning workspace..."
az ml workspace create -n $WORKSPACE -g $GROUP -l $LOCATION

echo "Configuring Azure CLI defaults..."
az configure --defaults group=$GROUP workspace=$WORKSPACE location=$LOCATION

echo "Setting up workspace..."
bash -x setup-workspace.sh

echo "Setting up extra workspaces..."
bash -x create-workspace-extras.sh
