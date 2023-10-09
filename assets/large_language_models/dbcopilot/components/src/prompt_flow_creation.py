# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The component create the example promptflow for db copilot."""
import functools
import json
import logging
import os
from typing import Dict

from component_base import ComponentBase, main_entry_point


@main_entry_point("create")
class PromptFlowCreation(ComponentBase):
    """PromptFlowCreation Class."""

    FLOW_JSON = os.path.join(
        os.path.dirname(__file__), "prompt_flows", "db_copilot_flow.json"
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

    @functools.cached_property
    def prompt_flow_url(self) -> str:
        """Get the prompt flow url."""
        return (
            self.service_endpoint
            + "/studioservice/api"
            + self.workspace_scope
            + "/flows"
        )

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
    ):
        """Create the prompt flow."""
        from utils.asset_utils import get_datastore_uri, parse_connection
        from utils.requests_utils import request

        logging.info("Creating PromptFlow for DBCopilot")
        workspace = self.workspace

        # find datastore uri
        datastore_uri = get_datastore_uri(workspace, asset_uri)
        logging.info(f"Datastore uri: {datastore_uri}")

        # find embedding & chat connections
        embedding_connection_id = ""
        chat_connection_id = ""

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

        # create flow
        with open(self.FLOW_JSON, "r") as f:
            flow_string = f.read()
            flow_string = flow_string.replace(
                "@@EMBEDDING_CONNECTION@@", embedding_aoai_connection
            )
            # TODO: add chat connection into tool
            flow_string = flow_string.replace(
                "@@CHAT_CONNECTION@@", chat_aoai_connection
            )
            flow_string = flow_string.replace("@@FLOW_NAME@@", index_name + "_Flow")
            flow_string = flow_string.replace(
                "@@EMBEDDING_AOAI_DEPLOYMENT_NAME@@", embedding_aoai_deployment_name
            )
            flow_string = flow_string.replace(
                "@@CHAT_AOAI_DEPLOYMENT_NAME@@",
                chat_aoai_deployment_name if chat_aoai_deployment_name else "",
            )
            flow_string = flow_string.replace(
                "@@GROUNDING_EMBEDDING_URI@@", grounding_embedding_uri
            )
            flow_string = flow_string.replace("@@DB_CONTEXT_URI@@", db_context_uri)
            flow_string = flow_string.replace("@@DATASTORE_URI@@", datastore_uri)
            flow_string = flow_string.replace("@@EXAMPLE_EMBEDDING_URI@@", example_embedding_uri
                                              if example_embedding_uri else "")
            flow_json = json.loads(flow_string)
        response = request(
            "post", self.prompt_flow_url, json=flow_json, headers=self.default_headers
        )
        pf_response_json = json.loads(response.text)
        flow_id = pf_response_json["flowResourceId"]
        parent_run = self.run.parent
        while parent_run:
            parent_run.add_properties({"azureml.promptFlowResourceId": flow_id})
            parent_run = parent_run.parent
