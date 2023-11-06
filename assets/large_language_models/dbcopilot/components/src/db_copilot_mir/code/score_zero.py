# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""File for MIR endpoint score function."""
import json
import logging
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from db_copilot.contract.memory_core import MemoryItem
from db_copilot_tool.contracts.db_copilot_config import DBCopilotConfig
from db_copilot_tool.history_service.dialogue_sessions import DialogueSession
from db_copilot_tool.history_service.history_service import (
    HistoryService,
    HistoryServiceConfig,
)
from db_copilot_tool.telemetry import set_print_logger
from db_copilot_tool.tools.azureml_asset_handler import DatastoreUploader
from db_copilot_tool.tools.db_copilot_adapter import DBCopilotAdapter
from db_copilot_tool.tools.memory_cache import MemoryCache
from flask import Response, stream_with_context
from promptflow.connections import AzureOpenAIConnection


def init():
    """Initialize the class."""
    set_print_logger()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cache_dir = os.getenv("AZUREML_MODEL_DIR")
    global embedding_aoai_connection, chat_aoai_connection, shared_config, history_service
    global db_copilots
    global db_copilots_shared
    db_copilots = MemoryCache(60 * 60)
    db_copilots_shared = MemoryCache(24 * 30 * 60 * 60)
    # setting up history service
    history_service_cache_dir = os.path.join(tempfile.tempdir, "history_cache")
    if not os.path.exists(history_service_cache_dir):
        os.makedirs(history_service_cache_dir)
    history_service_config = HistoryServiceConfig(
        history_service_enabled=True,
        cache_dir=history_service_cache_dir,
        expire_seconds=3600,
        max_cache_size_mb=100,
    )
    history_service = HistoryService(history_service_config)
    # setting up db copilot
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
    if cache_dir:
        for visibility in [Visibility.Private, Visibility.Shared]:
            folder_path = os.path.join(cache_dir, visibility.value.lower())
            if os.path.exists(folder_path):
                for db_name in os.listdir(folder_path):
                    if os.path.exists(
                        os.path.join(folder_path, db_name, "config.json")
                    ):
                        with open(
                            os.path.join(folder_path, db_name, "config.json"), "r"
                        ) as f:
                            logging.info(f"Loading config for {db_name}")
                            config_json = json.load(f)
                            config = DBCopilotConfig(**config_json)
                            set_db_copilot_adapter(
                                db_name, config, visibility=visibility
                            )

    config_file = os.path.join(current_dir, "configs.json")
    if os.path.exists(config_file):
        with open(config_file) as f:
            configs = json.load(f)
            if isinstance(configs, list):
                for config in configs:
                    name = config.get("db_name", None) or config.get(
                        "datastore_uri", None
                    )
                    if "db_name" in config:
                        config.pop("db_name", None)
                    config = {**shared_config, **config}
                    db_copilot_config = DBCopilotConfig(**config)
                    set_db_copilot_adapter(
                        name, db_copilot_config, visibility=Visibility.Shared
                    )
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


def set_db_copilot_adapter(db_name, db_copilot_config, visibility):
    cache_dir = os.getenv("AZUREML_MODEL_DIR")
    cache_uri = os.getenv("DBCOPILOT_CACHE_URI")

    def simple_db_name(db_name):
        if db_name.startswith("azureml://"):
            return db_name.split("/")[-1].replace(".", "_")
        return db_name

    if cache_uri:
        cache_folder = os.path.join(
            cache_dir, visibility.value.lower(), simple_db_name(db_name)
        )
        with DatastoreUploader(
            f"{cache_uri}{visibility.value.lower()}/{simple_db_name(db_name)}",
            cache_folder,
        ):
            db_copilot_adapter = DBCopilotAdapter(
                db_copilot_config,
                embedding_aoai_connection=embedding_aoai_connection,
                chat_aoai_connection=chat_aoai_connection,
                history_service=history_service,
                cache_folder=cache_folder,
            )
    else:
        db_copilot_adapter = DBCopilotAdapter(
            db_copilot_config,
            embedding_aoai_connection=embedding_aoai_connection,
            chat_aoai_connection=chat_aoai_connection,
            history_service=history_service,
        )
    if visibility == Visibility.Shared:
        db_copilots_shared.set(db_name, db_copilot_adapter)
    else:
        db_copilots.set(db_name, db_copilot_adapter)


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
    SampleQueries = "SampleQueries"
    Summary = "Summary"


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
    # history memory
    history_memory: Optional[List[MemoryItem]] = None
    include_memory: bool = False
    include_sample_queries: bool = False
    # for dbcopilot
    extra_kwargs: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Post init."""
        if isinstance(self.request_type, str):
            self.request_type = RequestType(self.request_type)
        if isinstance(self.visibility, str):
            self.visibility = Visibility(self.visibility)
        if self.history_memory and not isinstance(self.history_memory, list):
            raise ValueError("history_memory must be a list")
        if self.history_memory:
            history_memory = []

            for memory_item in self.history_memory:
                if isinstance(memory_item, dict):
                    memory_item = MemoryItem.from_dict(memory_item)
                    history_memory.append(memory_item)
            self.history_memory = history_memory

    def to_db_copilot_config(self, shared_config: dict):
        """To db copilot config."""
        config_dict = asdict(self)
        block_keys = ["request_type", "visibility", "db_name", "question"]
        for key in block_keys:
            config_dict.pop(key, None)
        config_dict = {**shared_config, **config_dict}
        return DBCopilotConfig(**config_dict)


def _get_db_provider(request_body: RequestBody, session_id: str):
    db_copilot = None
    if request_body.db_name or request_body.datastore_uri:
        db_copilot: DBCopilotAdapter = db_copilots_shared.get(
            request_body.db_name
            if request_body.db_name
            else request_body.datastore_uri,
            None,
        )
        return db_copilot
    elif session_id:
        db_copilot: DBCopilotAdapter = db_copilots.get(session_id, None)
        if db_copilot:
            return db_copilot
    if len(db_copilots_shared.get_all()) == 1:
        db_copilot: DBCopilotAdapter = list(db_copilots_shared.get_all().values())[0]
        return db_copilot
    else:
        return AMLResponse(
            "No db_copilot is available. Please specify Session id or datasotre_uri or db_name is required",
            400,
        )


@rawhttp
def run(request: AMLRequest):
    """run."""
    session_id = request.headers.get("session-id", None)
    if not session_id:
        session_id = request.headers.get("aml-session-id", None)
    request_id = request.headers.get("x-request-id", None)
    logging.info("Request session id: %s", session_id)
    logging.info("Request id: %s", request_id)
    cors_headers = {
        "Allow": "OPTIONS, GET, POST",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Methods": "OPTIONS, GET, POST",
    }
    try:
        request_body = RequestBody(**request.json)
        logging.info("Request body: %s", request_body)
        if request_body.history_memory and not session_id:
            session_id = str(uuid.uuid4())
        if request_body.history_memory:
            logging.info("Request session id: %s", session_id)
            logging.info("Request id: %s", request_id)
            dialogue_session = DialogueSession(request_body.history_memory)
            history_service.set_dialogue_session(
                session_id, dialogue_session, expire_seconds=60 * 5
            )
        if request_body.request_type == RequestType.Summary:
            db_copilot = _get_db_provider(request_body, session_id)
            if isinstance(db_copilot, AMLResponse):  # failed to get db_copilot
                return db_copilot
            summary, schema = db_copilot.summary
            return AMLResponse(
                {
                    "summary": summary,
                    "schema": schema,
                },
                json_str=True,
                status_code=200,
            )
        if request_body.request_type == RequestType.Grounding:
            logging.info("Grounding request")
            if request_body.datastore_uri:
                db_copilot_config = request_body.to_db_copilot_config(shared_config)
                db_name = (
                    (
                        request_body.db_name
                        if request_body.db_name
                        else request_body.datastore_uri
                    )
                    if request_body.visibility == Visibility.Shared
                    else session_id
                )
                set_db_copilot_adapter(
                    db_name, db_copilot_config, request_body.visibility
                )
            else:
                raise Exception("datastore_uri is required")
        if request_body.request_type == RequestType.Chat or (
            request_body.request_type == RequestType.Grounding and request_body.question
        ):
            logging.info("Chat request")

            if not request_body.question:
                return AMLResponse("Question is required", 400)
            db_copilot = _get_db_provider(request_body, session_id)
            if isinstance(db_copilot, AMLResponse):  # failed to get db_copilot
                return db_copilot
            if db_copilot is None:
                response = AMLResponse(
                    f"No db_copilot is available for session {session_id}", 400
                )
                for key, value in cors_headers.items():
                    response.headers[key] = value
                return response
            if request_body.is_stream:
                return Response(
                    stream_with_context(
                        (
                            json.dumps(item)
                            for item in db_copilot.stream_generate(
                                request_body.question,
                                session_id,
                                request_body.temperature,
                                request_body.top_p,
                                request_body.extra_kwargs,
                            )
                        )
                    ),
                    headers=cors_headers,
                    mimetype="application/json",
                    status=200,
                )
            else:
                if request_body.include_sample_queries or request_body.include_memory:
                    response = AMLResponse(
                        {
                            "response": list(
                                db_copilot.generate(
                                    request_body.question,
                                    session_id,
                                    request_body.temperature,
                                    request_body.top_p,
                                    request_body.extra_kwargs,
                                )
                            ),
                            "sample_queries": db_copilot.get_example_queries(
                                request_body.question, session_id
                            )
                            if request_body.include_sample_queries
                            else [],
                            "history_memory": [
                                asdict(memory)
                                for memory in history_service.get_dialogue_session(
                                    session_id
                                ).messageHistory
                            ]
                            if request_body.include_memory
                            else None,
                        },
                        json_str=True,
                        status_code=200,
                    )
                else:
                    response = AMLResponse(
                        list(
                            db_copilot.generate(
                                request_body.question,
                                session_id,
                                request_body.temperature,
                                request_body.top_p,
                                request_body.extra_kwargs,
                            )
                        ),
                        json_str=True,
                        status_code=200,
                    )
                for key, value in cors_headers.items():
                    response.headers[key] = value
                return response
        elif request_body.request_type == RequestType.GetSources:
            logging.info("GetSources request")
            sources = db_copilots_shared.get_all().keys()
            logging.info("Sources: %s", sources)
            response = AMLResponse(list(sources), json_str=True, status_code=200)
            for key, value in cors_headers.items():
                response.headers[key] = value
            return response
        elif request_body.request_type == RequestType.SampleQueries:
            logging.info("SampleQueries request")
            if not request_body.db_name:
                return AMLResponse("db_name is required", 400)
            if not request_body.question:
                return AMLResponse("Question is required", 400)
            db_copilot = _get_db_provider(request_body, session_id)
            return AMLResponse(
                db_copilot.get_example_queries(request_body.question, session_id),
                json_str=True,
                status_code=200,
            )

    except Exception as e:
        logging.exception("Exception: %s", e)
        response = AMLResponse(str(e), 500)
        for key, value in cors_headers.items():
            response.headers[key] = value
        return response
