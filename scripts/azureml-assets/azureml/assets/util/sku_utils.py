# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SKU utils."""

import json
import requests
from azureml.assets.util import logger
from azureml.assets.util.util import retry
from azure.identity import AzureCliCredential


all_sku_details = None
SKU_DETAILS_URI = (
    "https://management.azure.com/subscriptions/{}/"
    "providers/Microsoft.MachineLearningServices/locations/{}"
    "/vmSizes?api-version=2021-01-01&expandChildren=true"
)


@retry(3)
def get_all_sku_details(credential: AzureCliCredential, subscription_id: str, location: str = "eastus"):
    """Return all sku details.

    return response =>
    {
        "Standard_A1_v2": {
            "name": "Standard_A1_v2",
            "family": "standardAv2Family",
            "vCPUs": 1,
            "gpus": 0,
            "osVhdSizeMB": 1047552,
            "maxResourceVolumeMB": 10240,
            "memoryGB": 2.0,
            ....
        },
        ....
        ....
    }

    Args:
        credential (AzureCliCredential): Credential to generate token for the request
        subscription_id (str): Subscription ID to check details in
        location (str): location to query SKU details for. Default is set to eastus
    """
    global all_sku_details
    if all_sku_details is None:
        vmSizes = SKU_DETAILS_URI.format(subscription_id, location)
        token = credential.get_token("https://management.azure.com/.default")
        headers = {"Authorization": f"Bearer {token.token}"}
        response = requests.get(vmSizes, headers=headers)
        status_code = response.status_code
        content = response.content
        if status_code != 200:
            raise Exception(f"Unsuccessful requst. Response : {response}")
        sku_details_list = json.loads(content).get("amlCompute", [])
        all_sku_details = {sku_details["name"]: sku_details for sku_details in sku_details_list}

    return all_sku_details


def get_sku_details(credential: AzureCliCredential, SKU: str, subscription_id: str, location: str = "eastus"):
    """Get sku details.

    Args:
        credential (AzureCliCredential): Credential to generate token for the request
        SKU (str): SKU to fetch detail of
        subscription_id (str): Subscription ID to check details in
        location (str): location to query SKU details for. Default is set to eastus
    """
    global all_sku_details
    if all_sku_details is None:
        logger.print(f"Fetching all sku details for subscription: {subscription_id} and location: {location}")
        all_sku_details = get_all_sku_details(
            credential,
            subscription_id,
            location,
        )

    return all_sku_details.get(SKU, None)
