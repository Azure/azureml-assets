from promptflow.core import AzureOpenAIModelConfiguration
from promptflow.entities import AzureOpenAIConnection
from promptflow.client import PFClient
from azure.identity import DefaultAzureCredential
import os


def run_gpt_based_evaluator(connection, evaluator, inputs, deployment_name):
    model_config = AzureOpenAIModelConfiguration(connection=connection, azure_deployment=deployment_name)
    eval_fn = evaluator(model_config)
    return eval_fn(**inputs)
