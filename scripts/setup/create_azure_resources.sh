# Use of | tr -d '\n\r' is to handle newline mismatches on Windows systems

location="eastus"
resource_group="azureml-assets-static"
managed_identity="azureml-assets-uai"
federated_identity="azureml-assets"

echo "Retrieving subscription information..."
subscription_name=$(az account show --query name -o tsv | tr -d '\n\r')
subscription_id=$(az account show --query id -o tsv | tr -d '\n\r')

cat <<END
Please verify that resources should be created in this subscription:
${subscription_name} (${subscription_id})

If not, use the following commands to set the correct subscription:

    az login -t <tenant_id>
    az account set -s <subscription_id>

END
[[ "$(read -e -p 'Continue? [y/N] '; echo $REPLY)" == [Yy]* ]] || exit
echo

echo "Creating ${resource_group} resource group..."
az group show --name $resource_group || az group create --location $location --name $resource_group --tags "SkipAutoDeleteTill=2099-12-31"

echo "Creating ${managed_identity} user-assigned managed identity..."
az identity create --name $managed_identity --resource-group $resource_group --location $location --subscription $subscription_id  
managed_identity_id=$(az identity show --name $managed_identity --resource-group $resource_group --query principalId -o tsv | tr -d '\n\r')

echo "Creating ${federated_identity} federated identity credential..."
az identity federated-credential create --name $federated_identity --identity-name $managed_identity --resource-group $resource_group --issuer 'https://token.actions.githubusercontent.com' --subject 'repo:Azure/azureml-assets:environment:Testing' --audiences 'api://AzureADTokenExchange'

echo "Granting ${managed_identity} access to subscription..."
az role assignment create --role Contributor --assignee $managed_identity_id --scope /subscriptions/$subscription_id
