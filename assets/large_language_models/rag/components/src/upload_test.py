import os
import uuid
from pathlib import Path

from azureml.core import Workspace
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

CODE_DIR = "rag_code_flow"

workspace = Workspace.from_config()
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=workspace.subscription_id,
    resource_group_name=workspace.resource_group,
    workspace_name=workspace.name,
)

working_directory = ml_client.datastores.get("workspaceworkingdirectory")
asset_id = str(uuid.uuid4())
dest = f"index-in-pf-examples/Promptflows/{asset_id}/{CODE_DIR}"


def upload_code_files(ws: Workspace):
    """Upload the files in the code flow directory."""
    from azureml.data.dataset_factory import FileDatasetFactory

    working_directory = ws.datastores.get("workspaceworkingdirectory")
    asset_id = str(uuid.uuid4())
    dest = f"index-in-pf-examples/Promptflows/{asset_id}/{CODE_DIR}"
    FileDatasetFactory.upload_directory(
        os.path.join(Path(__file__).parent.absolute(), CODE_DIR),
        (working_directory, dest),
    )

    return dest


upload_code_files(workspace)
