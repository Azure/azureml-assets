# %%

from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient

import json

# Although this in principle is to be used as a script,
# it should also be usable inside an interactive notebook, for quick experimenting
# so we need to be able to switch between the two modes
# here we check if we are running in a notebook
_NOTEBOOK_MODE = False
try:
    get_ipython()
    _NOTEBOOK_MODE = True
except NameError:
    pass

if _NOTEBOOK_MODE:
    print("Script running in notebook mode")
# %%

credential = DefaultAzureCredential()
subscription_id = None
resource_group_name = None
if _NOTEBOOK_MODE:
    # edit here for the subscription id and resource group name
    subscription_id = "TBD"
    resource_group_name = "TBD"
else:
    # we will get the subscription id and resource group name
    # from the command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='List OpenAI resources and their deployments') 
    parser.add_argument('subscription_id', required=True, type=str, help='Azure subscription id')
    parser.add_argument('resource_group_name', required=True, type=str, help='Azure resource group name')  
    args = parser.parse_args()
    subscription_id = args.subscription_id
    resource_group_name = args.resource_group_name


print(f"subscription_id: {subscription_id}")
print(f"resource_group_name: {resource_group_name}")

resource_client = ResourceManagementClient(credential, subscription_id=subscription_id)

resource_type = "Microsoft.CognitiveServices/accounts"  # example resource type
account_kind = 'OpenAI'

# %%
print_details = True

resources = list(
    resource_client.resources.list_by_resource_group(
        resource_group_name,
        filter=f"resourceType eq '{resource_type}'"))
# retain  only kind == 'OpenAI'
resources = [r for r in resources if r.kind == account_kind]

# %%
# define a function to list deployments for a OpenAi endpoint 0
import requests

auth = credential.get_token("https://management.azure.com/.default")
auth_token = auth.token

def send_rest_request(url, method, headers, body):
    headers["Authorization"] = f"Bearer {auth_token}"
    headers["Content-Type"] = "application/json"
    response = requests.request(method, url, headers=headers, data=body)
    if response.status_code == 200:
        return response.json()    
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        requests.raise_for_status(response)

def list_deployments(account_name):
    uri = f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.CognitiveServices/accounts/{account_name}/deployments?api-version=2023-05-01"
    response = send_rest_request(uri, "GET", {}, "")
    # if it has "value property return that instead"
    if "keys" in dir(response) and "value" in response.keys():
        return response["value"]
    return response

def extract_essential(deployments):
    def trim_properties(properties):
        return {
            "model":properties["model"],
            "provisioningState":properties["provisioningState"],
            "capabilities":properties["capabilities"],
        }
    return [{
        "id": deployment["id"],
        "name": deployment["name"],
        "sku": deployment['sku'],
        "properties": trim_properties(deployment['properties']),
        "systemData": deployment['systemData']
    } for deployment in deployments]
# %%
loaded_resources = [
    resource_client.resources.get_by_id(r.id, "2021-04-30") 
    for r in resources
    ]
for r in loaded_resources:
    if r.properties['provisioningState'] != 'Succeeded':
        print(f"**** Resource {r.name} is not provisioned yet. Skipping.")
        continue
    else:
        print(r.name)
    provisioned = r.properties['provisioningState'] == 'Succeeded'   

    if (provisioned):
        deployments = list_deployments(r.name)
        if (deployments is not None and len(deployments) > 0):
            print(f"OAI Studio URLs: ")
            print(f" deployments URL - https://oai.azure.com/portal/{r.properties['internalId']}/deployment")
            print(f" quota (at subscription level) - https://oai.azure.com/portal/{r.properties['internalId']}/quota")
            print("~~~~~~~~ BEGIN deployments:")
            print(json.dumps(extract_essential(deployments), indent=2))
            print("~~~~~~~~ END deployments.")
    else:
        print(f"**** Skipping {r.name} -- as it has no deployment.")
    
# %%