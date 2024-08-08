from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.assets import publish_utils
from azureml.assets import config
from pathlib import Path

subscription_id = "4f26493f-21d2-4726-92ea-1ddd550b1d27"
resource_group = "registry-builtin-dev-eastus"
registry_name = "azureml-dev"

ml_client = MLClient(subscription_id=subscription_id, resource_group_name=resource_group, registry_name=registry_name, credential=DefaultAzureCredential())

path = Path('C:/Users/alisoy/source/repos/azureml-assets/assets/models/system/test-filetags-model/asset.yaml')

asset_config = config.AssetConfig(path)

publish_utils.create_asset(asset_config, registry_name, ml_client, debug=True)
# publish_utils.update_asset_metadata(asset_config, ml_client)