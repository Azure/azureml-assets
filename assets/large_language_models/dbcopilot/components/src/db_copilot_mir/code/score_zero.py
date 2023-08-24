# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""File for MIR endpoint score function."""
import json
import logging
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from db_copilot_tool.contracts.db_copilot_config import DBCopilotConfig
from db_copilot_tool.history_service.history_service import HistoryService
from db_copilot_tool.telemetry import set_print_logger
from db_copilot_tool.tools.db_copilot_adapter import DBCopilotAdapter
from db_copilot_tool.tools.memory_cache import MemoryCache
from flask import Response, stream_with_context
from promptflow.connections import AzureOpenAIConnection


def init():
    """Initialize the class."""
    set_print_logger()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    global embedding_aoai_connection, chat_aoai_connection, shared_config, history_service
    global db_copilots
    global db_copilots_shared
    db_copilots = MemoryCache(60 * 60)
    db_copilots_shared = MemoryCache(24 * 30 * 60 * 60)
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
        embedding_deploy_name = secret_manager.get("embedding-deploy-name")
        chat_deploy_name = secret_manager.get("chat-deploy-name")
    shared_config_file = os.path.join(current_dir, "shared_config.json")
    if os.path.exists(shared_config_file):
        with open(shared_config_file) as f:
            shared_config = json.load(f)
    else:
        shared_config = {}
    shared_config["embedding_aoai_deployment_name"] = embedding_deploy_name
    shared_config["chat_aoai_deployment_name"] = chat_deploy_name
    assert isinstance(shared_config, dict)
    config_file = os.path.join(current_dir, "configs.json")
    if os.path.exists(config_file):
        with open(config_file) as f:
            configs = json.load(f)
            if isinstance(configs, list):
                for config in configs:
                    name = config.get("db_name", None) or config.get("db_name", None)
                    if "db_name" in config:
                        config.pop("db_name", None)
                    config = {**shared_config, **config}
                    db_copilot_config = DBCopilotConfig(**config)
                    # TODO: fix history service sharing issue
                    db_copilot_adapter = DBCopilotAdapter(
                        db_copilot_config,
                        embedding_aoai_connection=embedding_aoai_connection,
                        chat_aoai_connection=chat_aoai_connection,
                        history_service=HistoryService(db_copilot_config.history_service_config),
                    )
                    # trigger grounding
                    db_copilot_adapter.db_provider_service
                    db_copilots_shared.set(name, db_copilot_adapter)
            else:
                raise ValueError(f"Invalid config file: {configs}")


def cleanup():
    """Clean up db_copilots cache."""
    if db_copilots:
        logging.info("Cleaning up db_copilots")
        db_copilots.delete()
    if db_copilots_shared:
        logging.info("Cleaning up db_copilots_shared")
        db_copilots_shared.delete()


class CaseInsensitiveEnum(Enum):
    """CaseInsensitiveEnum."""

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for member in cls:
                if member.name.lower() == value.lower():
                    return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")


class RequestType(CaseInsensitiveEnum):
    """RequestType."""

    Chat = "Chat"
    Grounding = "Grounding"
    GetSources = "GetSources"


class Visibility(CaseInsensitiveEnum):
    """Visibility."""

    Shared = "Shared"
    Private = "Private"


@dataclass
class RequestBody:
    """RequestBody."""

    request_type: Union[RequestType, str] = RequestType.Chat
    visibility: Union[Visibility, str] = Visibility.Shared
    db_name: Optional[str] = None
    question: Optional[str] = None
    datastore_uri: Optional[str] = None
    # LLM Config
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    # DBExecutorConfig
    selected_tables: List[str] = None
    column_settings: Dict[str, Dict[str, str]] = None
    # GroundingConfig
    max_tables: Optional[int] = None
    max_columns: Optional[int] = None
    max_rows: Optional[int] = None
    max_sampling_rows: Optional[int] = None
    max_text_length: Optional[int] = None
    is_stream: bool = False
    # Context
    tools: Optional[List[str]] = None
    include_built_in: bool = True
    sample_folder: Optional[str] = None
    knowledge_pieces: Optional[str] = None
    # index & context
    grounding_embedding_uri: str = None
    example_embedding_uri: str = None
    db_context_uri: str = None

    def __post_init__(self):
        """Post init."""
        if isinstance(self.request_type, str):
            self.request_type = RequestType(self.request_type)
        if isinstance(self.visibility, str):
            self.visibility = Visibility(self.visibility)

    def to_db_copilot_config(self, shared_config: dict):
        """To db copilot config."""
        config_dict = asdict(self)
        block_keys = ["request_type", "visibility", "db_name", "question"]
        for key in block_keys:
            config_dict.pop(key, None)
        config_dict = {**shared_config, **config_dict}
        return DBCopilotConfig(**config_dict)


@rawhttp
def run(request: AMLRequest):
    """run."""
    session_id = request.headers.get("session-id", None)
    if not session_id:
        session_id = request.headers.get("aml-session-id", None)
    request_id = request.headers.get("x-request-id", None)
    logging.info("Request session id: %s", session_id)
    logging.info("Request id: %s", request_id)

    try:
        request_body = RequestBody(**request.json)
        logging.info("Request body: %s", request_body)

        if request_body.request_type == RequestType.Grounding:
            logging.info("Grounding request")
            if request_body.datastore_uri:
                db_copilot_config = request_body.to_db_copilot_config(shared_config)
                db_copilot = DBCopilotAdapter(
                    db_copilot_config,
                    embedding_aoai_connection=embedding_aoai_connection,
                    chat_aoai_connection=chat_aoai_connection,
                    history_service=HistoryService(db_copilot_config.history_service_config),
                )
                if request_body.visibility == Visibility.Shared:
                    db_copilots_shared.set(
                        request_body.db_name if request_body.db_name else request_body.datastore_uri, db_copilot
                    )
                else:
                    db_copilots.set(session_id, db_copilot)
            else:
                raise Exception("datastore_uri is required")
        if request_body.request_type == RequestType.Chat or (
            request_body.request_type == RequestType.Grounding and request_body.question
        ):
            logging.info("Chat request")
            if not session_id and not request_body.datastore_uri and not request_body.db_name:
                return AMLResponse("Session id or datasotre_uri or db_name is required", 400)
            if not request_body.question:
                return AMLResponse("Question is required", 400)
            if request_body.db_name or request_body.datastore_uri:
                db_copilot: DBCopilotAdapter = db_copilots_shared.get(
                    request_body.db_name if request_body.db_name else request_body.datastore_uri, None
                )
            else:
                db_copilot: DBCopilotAdapter = db_copilots.get(session_id, None)
            if db_copilot is None:
                return AMLResponse(f"No db_copilot is available for session {session_id}", 400)
            if request_body.is_stream:
                return Response(
                    stream_with_context(
                        (
                            json.dumps(item)
                            for item in db_copilot.stream_generate(
                                request_body.question, session_id, request_body.temperature, request_body.top_p
                            )
                        )
                    ),
                    mimetype="application/json",
                    status=200,
                )
            else:
                return AMLResponse(
                    list(
                        db_copilot.generate(
                            request_body.question, session_id, request_body.temperature, request_body.top_p
                        )
                    ),
                    json_str=True,
                    status_code=200,
                )
        elif request_body.request_type == RequestType.GetSources:
            logging.info("GetSources request")
            sources = db_copilots_shared.get_all().keys()
            logging.info("Sources: %s", sources)
            return AMLResponse(list(sources), json_str=True, status_code=200)

    except Exception as e:
        logging.exception("Exception: %s", e)
        return AMLResponse(str(e), 500)
