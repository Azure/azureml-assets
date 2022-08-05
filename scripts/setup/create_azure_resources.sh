# Use of | tr -d '\n\r' is to handle newline mismatches on Windows systems

# Globals
APP_NAME="azureml-assets"
CREDENTIAL_PARAMS_FILE="credential_params.json"

echo "Retrieving subscription information..."
SUBSCRIPTION_NAME=$(az account show --query name -o tsv | tr -d '\n\r')
SUBSCRIPTION_ID=$(az account show --query id -o tsv | tr -d '\n\r')

cat <<END
Please verify that resources should be created in this subscription:
${SUBSCRIPTION_NAME} (${SUBSCRIPTION_ID})

If not, use the following commands to set the correct subscription:

    az login -t <tenant_id>
    az account set -s <subscription_id>

END
[[ "$(read -e -p 'Continue? [y/N] '; echo $REPLY)" == [Yy]* ]] || exit
echo

echo "Creating ${APP_NAME} application and service principal..."
az ad app create --display-name $APP_NAME
APP_ID=$(az ad app list --display-name "${APP_NAME}" --query "[0].appId" -o tsv | tr -d '\n\r')
az ad sp show --id $APP_ID || az ad sp create --id $APP_ID

echo "Adding federated credential to ${APP_NAME} application..."
az ad app federated-credential create --id $APP_ID --parameters $CREDENTIAL_PARAMS_FILE
exit

echo "Granting ${APP_NAME} access to subscription..."
az role assignment create --role Contributor --assignee $APP_ID
