import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential


subscription_id = "c0afea91-faba-4d71-bcb6-b08134f69982"
resource_group = "batchscore-test-centralus"
workspace = "ws-batchscore-centralus"

_current_file_path = Path(os.path.abspath(__file__))
configs_root = _current_file_path.parent / "configuration_files" / "for_e2e_tests"
configuration_files = {
    "aoai_completion_configuration": {
        "data_asset_name": "aoai_completion_configuration",
        "data_asset_version": "5",
        "file_path": configs_root / "aoai_completion.json",
    },
    "aoai_chat_completion_configuration": {
        "data_asset_name": "aoai_chat_completion_configuration",
        "data_asset_version": "5",
        "file_path": configs_root / "aoai_chat_completion.json",
    },
    "aoai_embedding_configuration": {
        "data_asset_name": "aoai_embedding_configuration",
        "data_asset_version": "5",
        "file_path": configs_root / "aoai_embedding.json",
    },
    "mir_completion_configuration": {
        "data_asset_name": "mir_completion_configuration",
        "data_asset_version": "5",
        "file_path": configs_root / "mir_completion.json",
    },
    "serverless_completion_configuration": {
        "data_asset_name": "serverless_completion_configuration",
        "data_asset_version": "5",
        "file_path": configs_root / "serverless_completion.json",
    },
}

ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
for config in configuration_files.values():
    my_data = Data(
        path=config["file_path"],
        type=AssetTypes.URI_FILE,
        description="file-based config for e2e test",
        name=config["data_asset_name"],
        version=config["data_asset_version"],
    )
    ml_client.data.create_or_update(my_data)
