import os
from azure.ai.ml import MLClient
from azure.ai.ml import command, MpiDistribution
from azure.identity import DefaultAzureCredential

def test_that_tests_run():
	subscription_id = os.environ.get("sub_id")
	resource_group = os.environ.get("resource_group")
	workspace_name = os.environ.get("workspace")
	ml_client = MLClient(
		DefaultAzureCredential(), subscription_id, resource_group, workspace_name
	)
	assert ml_client.workspace_name == "registry-built-in-assets-test-ws"