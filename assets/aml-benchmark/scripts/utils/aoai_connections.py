# %% markdown
# # First we import what we need and parse command line arguments
#   - subscription_id
#   - resource_group_name
#
# # or for running in notebooks define the equivalent global variables
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
account_name = None
deployment_name = None
action = None
if _NOTEBOOK_MODE:
    # edit here for the subscription id and resource group name
    # edit here for the subscription id and resource group name
    subscription_id = "TBD replace here"
    resource_group_name = "TBD replace here"
else:
    # we will get the subscription id and resource group name
    # from the command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Utility to manage Azure OpenAI resources and their deployments')
    parser.add_argument('--subscription_id', required=True, type=str, help='Azure subscription id')
    parser.add_argument('--resource_group_name', required=True, type=str, help='Azure resource group name')  
    
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')

    parser_list = subparsers.add_parser('list-connections', help='Print AOAI resources and their quota usage')

    parser_delete = subparsers.add_parser('delete-deployment', help='Delete a particular deployment')
    parser_delete.add_argument('--account_name', required=True, type=str, help='AOAI account(endpoint) name')  
    parser_delete.add_argument('--deployment_name', required=True, type=str, help='AOAI deployment name')  

    args = parser.parse_args()
    action = args.action
    subscription_id = args.subscription_id
    resource_group_name = args.resource_group_name
    if (args.action == 'delete-deployment'):
        account_name = args.account_name
        deployment_name = args.deployment_name

print(f"action: {action} ", )
print(f"subscription_id: {subscription_id}")
print(f"resource_group_name: {resource_group_name}")
print(f"account_name: {account_name}")
print(f"deployment_name: {deployment_name}")



resource_type = "Microsoft.CognitiveServices/accounts"  # example resource type
account_kind = 'OpenAI'

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
        try:
            return response.json()
        except json.decoder.JSONDecodeError:
            return response.text
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        response.raise_for_status()

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
# functions to suport deletions

_api_version = '2023-05-01'
def _aoai_deployment_url(account_name, deployment_name) -> str:
    """Aoai deployment url."""
    return 'https://management.azure.com/subscriptions/'\
           f'{subscription_id}/resourceGroups/{resource_group_name}/'\
           f'providers/Microsoft.CognitiveServices/accounts/{account_name}/'\
           f'deployments/{deployment_name}?api-version=2023-05-01'

def _do_delete(account_name, deployment_name):
    send_rest_request(
        _aoai_deployment_url(account_name,deployment_name),
        "DELETE",
        {},
        ""
    )
    
# %%
if (action == 'list-connections'):
    print_details = True

    resource_client = ResourceManagementClient(credential, subscription_id=subscription_id)

    resources = [ r for r in ( 
            resource_client.resources.list_by_resource_group(
                resource_group_name,
                filter=f"resourceType eq '{resource_type}'"))
        ]
    # retain  only kind == 'OpenAI'
    resources = [r for r in resources if r.kind == account_kind]
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
elif _NOTEBOOK_MODE:
    print("Skipping list-connections cell because action is not list-connections and we are not running in notebook mode")    
# %%
    
# %%
if (action == 'delete-deployment'):
    print("Deleting deployment")
    _do_delete(account_name, deployment_name)