# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The component create the example promptflow for db copilot."""
import logging
import os
from typing import Dict
import yaml

from component_base import main_entry_point, OBOComponentBase
from promptflow.azure import PFClient


@main_entry_point("create")
class PromptFlowCreation(OBOComponentBase):
    """PromptFlowCreation Class."""

    FLOW_DIRECTORY = os.path.join(
        os.path.dirname(__file__), "prompt_flows"
    )

    FLOW_DAG = os.path.join(
        os.path.dirname(__file__), "prompt_flows", "flow.dag.yaml"
    )

    def __init__(self):
        """Initialize the class."""
        super().__init__()

    parameter_type_mapping: Dict[str, str] = {
        "grounding_embedding_uri": "uri_folder",
        "example_embedding_uri": "uri_folder",
        "db_context_uri": "uri_folder",
    }

    parameter_mode_mapping: Dict[str, str] = {
        "db_context_uri": "direct",
        "grounding_embedding_uri": "direct",
        "example_embedding_uri": "direct",
    }

    def create(
        self,
        index_name: str,
        grounding_embedding_uri: str,
        embedding_aoai_deployment_name: str,
        db_context_uri: str,
        asset_uri: str,
        chat_aoai_deployment_name: str = None,
        example_embedding_uri: str = None,
        llm_config: str = None,
        runtime: str = None,
    ):
        """Create the prompt flow."""
        from utils.asset_utils import get_datastore_uri, parse_connection

        logging.info("Creating PromptFlow for DBCopilot")
        workspace = self.workspace

        # find datastore uri
        datastore_uri = get_datastore_uri(workspace, asset_uri)
        logging.info(f"Datastore uri: {datastore_uri}")

        embedding_connection_id = os.environ.get(
            "AZUREML_WORKSPACE_CONNECTION_ID_AOAI_EMBEDDING", None
        )
        if embedding_connection_id is not None and embedding_connection_id != "":
            connection_dict = parse_connection(embedding_connection_id)
            if connection_dict:
                embedding_aoai_connection = connection_dict["connection_name"]
                logging.info(f"Embedding connection: {embedding_aoai_connection}")
            else:
                logging.warning(
                    f"Unable to parse connection string {embedding_connection_id}"
                )
        else:
            logging.error("Unable to find AOAI embedding connection")
            raise ValueError("Unable to find AOAI embedding connection")

        chat_connection_id = os.environ.get(
            "AZUREML_WORKSPACE_CONNECTION_ID_AOAI_CHAT", None
        )
        if chat_connection_id is not None and chat_connection_id != "":
            connection_dict = parse_connection(chat_connection_id)
            if connection_dict:
                chat_aoai_connection = connection_dict["connection_name"]
                logging.info(f"Chat connection: {chat_aoai_connection}")
            else:
                logging.warning(
                    f"Unable to parse connection string {chat_connection_id}"
                )
        else:
            logging.error("Unable to find AOAI chat connection")
            raise ValueError("Unable to find AOAI chat connection")

        if (
                chat_aoai_deployment_name is None
                and llm_config is not None
                and llm_config != ""
        ):
            chat_aoai_deployment_name = self.parse_llm_config(llm_config)

        # update flow
        with open(self.FLOW_DAG, "r", encoding="utf-8") as file:
            flow_data = yaml.safe_load(file)
            flow_data['name'] = index_name + "_Flow"
            flow_data['nodes'][0]['inputs']['embedding_aoai_config'] = embedding_aoai_connection
            flow_data['nodes'][0]['inputs']['chat_aoai_config'] = chat_aoai_connection
            flow_data['nodes'][0]['inputs']['db_context_uri'] = db_context_uri
            flow_data['nodes'][0]['inputs']['grounding_embedding_uri'] = grounding_embedding_uri
            flow_data['nodes'][0]['inputs']['datastore_uri'] = datastore_uri
            flow_data['nodes'][0]['inputs']['embedding_aoai_deployment_name'] = embedding_aoai_deployment_name
            flow_data['nodes'][0]['inputs'][
                'chat_aoai_deployment_name'] = chat_aoai_deployment_name if chat_aoai_deployment_name else ""
            flow_data['nodes'][0]['inputs']['example_embedding_uri'] = example_embedding_uri

        with open(self.FLOW_DAG, 'w', encoding="utf-8") as file:
            yaml.safe_dump(flow_data, file)

        # create flow
        pf_client = PFClient(ml_client=self.ml_client)
        flow = self.FLOW_DIRECTORY
        data = os.path.join(self.FLOW_DIRECTORY, "data.jsonl")
        environment_variables = {
            "AZUREML_WORKSPACE_NAME": self.workspace.name,
            "AZUREML_SUBSCRIPTION_ID": self.workspace.subscription_id,
            "AZUREML_RESOURCE_GROUP": self.workspace.resource_group,
        }
        base_run = pf_client.run(
            flow=flow,
            data=data,
            runtime=runtime,
            environment_variables=environment_variables,
        )
        pf_client.stream(base_run)
