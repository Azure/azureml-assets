# Globals
LOCATION="eastus"
DATESTAMP=$(date "+%Y%m%d")
RESOURCE_GROUP="azureml-assets-${DATESTAMP}"
WORKSPACE="azureml-assets-ws-${DATESTAMP}"

echo "Installing Azure CLI extension for Azure Machine Learning..."
az extension add -n ml -y

echo "Creating resource group ${RESOURCE_GROUP}..."
az group show -n $RESOURCE_GROUP || az group create -n $RESOURCE_GROUP -l $LOCATION

echo "Configuring Azure CLI defaults..."
az configure --defaults group=$RESOURCE_GROUP workspace=$WORKSPACE location=$LOCATION

echo "Creating Azure Machine Learning workspace..."
az ml workspace show -n $WORKSPACE | az ml workspace create

exit

echo "Setting up workspace..."
bash -x setup-workspace.sh

echo "Setting up extra workspaces..."
bash -x create-workspace-extras.sh

# TODO: Output resource group name, workspace name