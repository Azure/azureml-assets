from promptflow.core import AzureOpenAIModelConfiguration
from promptflow.entities import AzureOpenAIConnection
from promptflow.client import PFClient
from azure.identity import DefaultAzureCredential
import os


def run_gpt_based_evaluator(evaluator, inputs, deployment_name):
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
            api_type="azure",
        )
        conn = pf.connections.create_or_update(connection)
        print("connection created")

    # os.environ.pop("REQUESTS_CA_BUNDLE", None)
    # os.environ.pop("SSL_CERT_DIR", None)

    model_config = AzureOpenAIModelConfiguration(connection=conn.name, azure_deployment=deployment_name)
    eval_fn = evaluator(model_config)
    return eval_fn(**inputs)
