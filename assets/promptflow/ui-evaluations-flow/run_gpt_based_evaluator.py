from promptflow.core import AzureOpenAIModelConfiguration
from promptflow.entities import AzureOpenAIConnection
from promptflow.client import PFClient
from azure.identity import DefaultAzureCredential



def run_gpt_based_evaluator(evaluator, connection, inputs, deployment_name="gpt-4"):
    pf = PFClient(
        credential=DefaultAzureCredential(),
        subscription_id="",
        resource_group_name="",
        workspace_name="",
    )

    conn_name = "ui-evals-connection"
    try:
        conn = pf.connections.get(conn_name)
        print("connection already exists")
    except:
        connection = AzureOpenAIConnection(
            name=conn_name,
            api_key="",
            api_base="",
            api_type="",
        )
        conn = pf.connections.create_or_update(connection)
        print("connection created")

    model_config = AzureOpenAIModelConfiguration(
        connection=conn.name,
        azure_deployment="gpt-4",
    )

    model_config = AzureOpenAIModelConfiguration(connection=connection, azure_deployment=deployment_name)
    eval_fn = evaluator(model_config)
    return eval_fn(**inputs)
