# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Component for grounding database tables and columns."""
import json
import logging
from typing import Dict

from component_base import ComponentBase, main_entry_point
from db_copilot.db_provider.grounding.grounding_service import GroundingConfig
from db_copilot_tool.contracts.embedding_config import EmbeddingConfig
from db_copilot_tool.tools.db_executor_factory import DBExecutorConfig
from db_copilot_tool.tools.db_provider_adapter import DBProviderServiceAdapter, DBProviderServiceConfig
from db_copilot_tool.tools.dummy_embedding_service import DummyEmbeddingService


@main_entry_point("ground")
class DataBaseGrounding(ComponentBase):
    """DataBaseGrounding Class."""

    parameter_type_mapping: Dict[str, str] = {
        "output_chunk_file": "uri_folder",
        "output_ngram_file": "uri_folder",
        "output_grounding_context_file": "uri_folder",
    }

    def ground(
        self,
        output_chunk_file: str,
        output_grounding_context_file: str,
        asset_uri: str = None,
        max_tables: int = None,
        max_columns: int = None,
        max_rows: int = None,
        max_sampling_rows: int = None,
        max_text_length: int = None,
        selected_tables: str = None,
        column_settings: str = None,
        include_views: bool = False,
    ):
        """Ground database tables and columns."""
        from utils.asset_utils import get_datastore_uri

        logging.info("Grounding...")
        embedding_config = EmbeddingConfig()

        grounding_config = GroundingConfig()
        if max_tables:
            grounding_config.max_tables = max_tables
        if max_columns:
            grounding_config.max_columns = max_columns
        if max_rows:
            grounding_config.max_rows = max_rows
        if max_sampling_rows:
            grounding_config.max_sampling_rows = max_sampling_rows
        if max_text_length:
            grounding_config.max_text_length = max_text_length

        db_executor_config = DBExecutorConfig()
        if selected_tables:
            db_executor_config.tables = json.loads(selected_tables)
            if not isinstance(db_executor_config.tables, list):
                raise ValueError("selected_tables must be a list")
        if column_settings:
            db_executor_config.column_settings = json.loads(column_settings)
            if not isinstance(db_executor_config.column_settings, dict):
                raise ValueError("column_settings must be a dict")
        if include_views:
            if db_executor_config.metadata is None:
                db_executor_config.metadata = {}
            db_executor_config.metadata["include_views"] = include_views
        datastore_uri = get_datastore_uri(self.workspace, asset_uri)
        db_provider_config = DBProviderServiceConfig(
            db_uri=datastore_uri,
            embedding_service_config=embedding_config,
            grounding_config=grounding_config,
            db_executor_config=db_executor_config,
        )

        db_provider = DBProviderServiceAdapter(db_provider_config, workspace=self.workspace)
        assert isinstance(db_provider.embedding_service, DummyEmbeddingService)
        db_provider.embedding_service.dump(output_chunk_file)
        db_provider.dump_context(output_grounding_context_file)
