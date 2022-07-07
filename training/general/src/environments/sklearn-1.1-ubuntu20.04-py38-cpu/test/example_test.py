from azure.ai.ml import MLClient
from azure.ai.ml import command, MpiDistribution
from azure.identity import DefaultAzureCredential

def test_that_tests_run():
	subscription_id = workspace_sub_id
	resource_group = workspace_resource_group
	workspace = workspace_name
	ml_client = MLClient(
		DefaultAzureCredential(), subscription_id, resource_group, workspace
	)
	assert ml_client.workspace_name == "registry-built-in-assets-test-ws"