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
from db_copilot_tool.history_service.history_service import HistoryService
from db_copilot_tool.telemetry import set_print_logger
from db_copilot_tool.tools.db_copilot_adapter import DBCopilotAdapter
from flask import Response, stream_with_context
from promptflow.connections import AzureOpenAIConnection


def init():
    """Initialize the class."""
    global db_copilot
    set_print_logger()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(current_dir, "secrets.json")) as f:
        secret_manager: dict = json.load(f)
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
            db_copilot_config_dict: dict = json.load(f)
            logging.info("DBCopilot config: %s", db_copilot_config_dict)
            # validate config
            db_copilot_config = DBCopilotConfig(**db_copilot_config_dict)
            history_service = HistoryService(db_copilot_config.history_service_config)
            db_copilot = DBCopilotAdapter(
                db_copilot_config,
                embedding_aoai_connection=embedding_aoai_connection,
                chat_aoai_connection=chat_aoai_connection,
                history_service=history_service,
            )


def stream_generate(
    question: str,
    session_id: Optional[str],
    temperature: float = None,
    top_p: float = None,
):
    """generate."""
    for cells in db_copilot.stream_generate(question, session_id, temperature, top_p):
        logging.info("Cells: %s", cells)
        yield json.dumps(cells)


def generate(
    question: str,
    session_id: Optional[str],
    temperature: float = None,
    top_p: float = None,
):
    """generate."""
    cells = list(db_copilot.stream_generate(question, session_id, temperature, top_p))[
        -1
    ]
    logging.info("Cells: %s", cells)
    return json.dumps(cells)


@rawhttp
def run(request: AMLRequest):
    """run."""
    session_id = request.headers.get("session-id", None)
    if not session_id:
        session_id = request.headers.get("aml-session-id", None)
    request_id = request.headers.get("x-request-id", None)
    temperature = request.headers.get("llm-temperature", None)
    top_p = request.headers.get("llm-top-p", None)
    is_stream = request.headers.get("llm-stream", False)
    question = request.get_data(as_text=True)
    logging.info("Request question: %s", question)
    logging.info("Request session id: %s", session_id)
    logging.info("Request temperature: %s", temperature)
    logging.info("Request top p: %s", top_p)
    logging.info("Request is stream: %s", is_stream)
    logging.info("Request id: %s", request_id)
    try:
        if is_stream:
            return Response(
                stream_with_context(
                    (
                        json.dumps(item)
                        for item in db_copilot.stream_generate(
                            question, session_id, temperature, top_p
                        )
                    )
                ),
                mimetype="application/json",
                status=200,
            )
        else:
            return AMLResponse(
                list(db_copilot.generate(question, session_id, temperature, top_p)),
                json_str=True,
                status_code=200,
            )
    except Exception as e:
        logging.exception("Exception: %s", e)
        return AMLResponse(str(e), 500)
