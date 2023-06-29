"""The component generates the ground truth for the examples."""
import csv
import json
import logging
import os
from dataclasses import asdict
from typing import Dict

import pandas as pd
from component_base import ComponentBase, main_entry_point


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
        database_type: str = "sqlserver",
        tools: str = None,
        sample_folder: str = None,
        grounding_context: str = None,
    ):
        """ground_examples."""
        from db_copilot.contract.chat_core import InContextExample
        from db_copilot.contract.db_core import DatabaseType, SQLDialect
        from db_copilot.prompts.examples.base import get_examples

        if (
            grounding_context
            and os.path.exists(grounding_context)
            and os.path.exists(os.path.join(grounding_context, "db_type.json"))
        ):
            with open(os.path.join(grounding_context, "db_type.json"), "r") as f:
                grounding_database_type = json.load(f)
                if database_type and grounding_database_type != database_type:
                    logging.warn("grounding database type is different from input")
                database_type = grounding_database_type
        dialect = SQLDialect.from_db_type(DatabaseType(database_type))
        tools = json.loads(tools) if tools else None
        with_extensions = (
            True
            if tools
            and len([tool for tool in tools if tool.lower() != dialect.value]) > 0
            else False
        )
        examples = get_examples(dialect, with_extensions=with_extensions)
        if sample_folder:
            for file in os.listdir(sample_folder):
                try:
                    with open(os.path.join(sample_folder, file), "r") as example:
                        examples_obj = json.load(example)
                        if isinstance(examples_obj, list):
                            for example_obj in examples_obj:
                                example = InContextExample(**example_obj)
                                examples.append(example)
                        elif isinstance(examples_obj, dict):
                            example = InContextExample(**examples_obj)
                            examples.append(example)
                        else:
                            raise ValueError(
                                f"Invalid sample file format: {examples_obj}"
                            )
                except Exception as ex:
                    logging.warn(f"Failed to load sample file {file}. Error: {ex}")

        output_file = os.path.join(output_chunk_file, "Chunks_sample.csv")
        logging.info(f"output path is {output_file}")
        db_chunks_df = pd.DataFrame(columns=["Metadata", "Chunk"])
        # build embedding index
        for example in examples:
            db_chunks_df = pd.concat(
                [
                    db_chunks_df,
                    pd.DataFrame(
                        {
                            "Metadata": json.dumps(
                                {
                                    "example": asdict(example),
                                    "dialect": dialect,
                                    "source": {"filename": "example_file"},
                                }
                            ),
                            "Chunk": example.embed_text,
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
                axis=0,
            )  # type: ignore
        if not os.path.exists(output_chunk_file):
            os.makedirs(output_chunk_file)
        db_chunks_df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
        logging.info(f"saved chunk file to {output_file}")
