# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The component generates the ground truth for the examples."""
import json
import os
from typing import Dict

from component_base import ComponentBase, main_entry_point
from db_copilot.contract.db_core import DatabaseType
from db_copilot_tool.contracts.embedding_config import EmbeddingConfig
from db_copilot_tool.tools.dummy_embedding_service import DummyEmbeddingService
from db_copilot_tool.tools.in_context_learning_agent import InContextLearningAgent


@main_entry_point("ground")
class ExampleGrounding(ComponentBase):
    """ExampleGrounding Class."""

    def __init__(self) -> None:
        """Initialize the class."""
        super().__init__()

    parameter_type_mapping: Dict[str, str] = {
        "sample_folder": "uri_folder",
        "output_chunk_file": "uri_folder",
        "grounding_context": "uri_folder",
    }

    def ground(
        self,
        output_chunk_file: str,
        grounding_context: str,
        include_builtin: bool = True,
        tools: str = None,
        sample_folder: str = None,
    ):
        """Ground the examples."""
        if (
            grounding_context
            and os.path.exists(grounding_context)
            and os.path.exists(os.path.join(grounding_context, "db_type.json"))
        ):
            with open(os.path.join(grounding_context, "db_type.json"), "r") as f:
                database_type = json.load(f)
            db_type = DatabaseType(database_type)
            tools = json.loads(tools) if tools else None
            if tools:
                assert isinstance(tools, list) or isinstance(tools, dict)
            examples = InContextLearningAgent.get_examples(
                example_uri=sample_folder,
                db_type=db_type,
                tools=tools,
                include_built_in=include_builtin,
            )
            agent = InContextLearningAgent(
                embedding_config=EmbeddingConfig(), examples=examples
            )

            assert isinstance(agent.embedding_service, DummyEmbeddingService)
            agent.embedding_service.dump(output_chunk_file)
