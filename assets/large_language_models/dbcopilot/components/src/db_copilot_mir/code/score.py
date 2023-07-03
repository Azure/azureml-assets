# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""File for MIR endpoint score function."""
import json
import logging
import os
from typing import Optional

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from db_copilot_tool.contracts.db_copilot_config import DBCopilotConfig
from db_copilot_tool.db_copilot_tool import DBCopilot
from db_copilot_tool.telemetry import set_logger
from promptflow.connections import AzureOpenAIConnection
from promptflow.core.secret_manager import ConfigBasedSecretManager


def init():
    """Initialize the class."""
    global dbcopilot
    set_logger()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    secret_manager = ConfigBasedSecretManager(os.path.join(current_dir, "secrets.json"))
    embedding_aoai_connection = AzureOpenAIConnection(
        api_key=secret_manager.get("embedding-aoai-api-key"),
        api_base=secret_manager.get("embedding-aoai-api-base"),
        api_type="azure",
        api_version="2023-03-15-preview",
    )
    chat_aoai_connection = AzureOpenAIConnection(
        api_key=secret_manager.get("chat-aoai-api-key"),
        api_base=secret_manager.get("chat-aoai-api-base"),
        api_type="azure",
        api_version="2023-03-15-preview",
    )
    dbcopilot_config_file = os.path.join(current_dir, "db_copilot_config.json")
    with open(dbcopilot_config_file) as f:
        dbcopilot_config_dict: dict = json.load(f)
        logging.info("DBCopilot config: %s", dbcopilot_config_dict)
        # validate config
        dbcopilot_config = DBCopilotConfig(**dbcopilot_config_dict)
        dbcopilot = DBCopilot(
            embedding_aoai_config=embedding_aoai_connection,
            chat_aoai_config=chat_aoai_connection,
            **dbcopilot_config.to_db_copilot_dict(),
        )
        # dbcopilot.config.is_stream = True


def generate(question: str, session_id: Optional[str], temperature: float = 0.0):
    """generate."""
    for cell in dbcopilot.stream_generate(question, session_id, temperature):
        logging.info("Cell: %s", cell)
        yield json.dumps(cell)


@rawhttp
def run(request: AMLRequest):
    """run."""
    session_id = request.headers.get("session-id", None)
    if not session_id:
        session_id = request.headers.get("aml-session-id", None)
    request_id = request.headers.get("x-request-id", None)
    temperature = request.headers.get("llm-temperature", 0.0)
    question = request.get_data(as_text=True)
    logging.info("Request question: %s", question)
    logging.info("Request session id: %s", session_id)
    logging.info("Request temperature: %s", temperature)
    logging.info("Request id: %s", request_id)
    # return Response(
    #     stream_with_context(generate(question, session_id, temperature)),
    #     mimetype="application/json",
    #     status=200,
    # )
    return AMLResponse(list(generate(question, session_id, temperature)), 200)
