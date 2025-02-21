from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import azureml.assets as assets
from azureml.assets.publish_utils import create_asset
from azureml.assets.model.registry_utils import update_model_metadata
from pathlib import Path

# Libraries for get_token
from pathlib import Path
from azureml.assets.get_tokens import get_tokens
import json

# Libraries for testing model metadata updates/archiving
# from azureml.assets.model.registry_utils import update_model_metadata
# from azureml.assets.deployment_config import AssetVersionUpdate

# SET-UP
subscription_id = "ea4faa5b-5e44-4236-91f6-5483d5b17d14"
resource_group = "kellyl"
registry_name = "kelly-registry"
ml_client = MLClient(
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    registry_name=registry_name,
    credential=DefaultAzureCredential(),
)

# Set SAS token here for testing purposes
def set_sas_token(extra_config, my_asset_path):
    print(extra_config)
    storage_path: AzureBlobstoreAssetPath = extra_config.path
    print(storage_path)
    input_dirs = [Path(my_asset_path)]
    json_output_path = "tokens.json"
    get_tokens(input_dirs, "asset.yaml", json_output_path, 72)
    with open(json_output_path, 'r') as json_token_file:
        tokens_dict = json.load(json_token_file)
        token = tokens_dict[storage_path.storage_name][storage_path.container_name]
        print(token)
        extra_config.path.token = token

# CREATE ASSET
def create_my_asset_config(my_asset_path):
    print(f"{my_asset_path}/{assets.DEFAULT_ASSET_FILENAME}")
    my_asset = assets.AssetConfig(Path(my_asset_path) / assets.DEFAULT_ASSET_FILENAME)
    print(my_asset)
    return my_asset



my_asset_path = "./my-triton-model"
my_asset = create_my_asset_config(my_asset_path)
print(my_asset.extra_config_as_object())
set_sas_token(my_asset.extra_config_as_object(), my_asset_path)
create_asset(my_asset, registry_name, ml_client)
