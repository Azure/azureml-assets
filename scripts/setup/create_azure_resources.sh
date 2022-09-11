# Use of | tr -d '\n\r' is to handle newline mismatches on Windows systems

app_name="azureml-assets"
credential_params_file="credential_params.json"

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

echo "Creating ${app_name} application and service principal..."
az ad app create --display-name $app_name
app_id=$(az ad app list --display-name "${app_name}" --query "[0].appId" -o tsv | tr -d '\n\r')
az ad sp show --id $app_id || az ad sp create --id $app_id

echo "Adding federated credential to ${app_name} application..."
az ad app federated-credential create --id $app_id --parameters $credential_params_file
exit

echo "Granting ${app_name} access to subscription..."
az role assignment create --role Contributor --assignee $app_id
