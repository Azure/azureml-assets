"""Component for grounding database tables and columns."""
import json
import logging
from typing import Dict

from component_base import ComponentBase, main_entry_point
from db_copilot.db_provider.grounding import GroundingConfig
from db_copilot_tool.tools.grounding_service_manager import (
    DBExecutorConfig,
    GroundingServiceManager,
)


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
    ):
        """Ground database tables and columns."""
        logging.info("Grounding...")
        grounding_config = GroundingConfig()
        db_executor_config = DBExecutorConfig()
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
        if selected_tables:
            db_executor_config.tables = json.loads(selected_tables)
            if not isinstance(db_executor_config.tables, list):
                raise ValueError("selected_tables must be a list")
        if column_settings:
            db_executor_config.column_settings = json.loads(column_settings)
            if not isinstance(db_executor_config.column_settings, dict):
                raise ValueError("column_settings must be a dict")

        grounding_service_manager = GroundingServiceManager.from_uri(
            asset_uri,
            grounding_config,
            db_executor_config,
            self.workspace,
        )

        grounding_service_manager.dump_chunk(output_chunk_file)
        grounding_service_manager.dump(output_grounding_context_file)
