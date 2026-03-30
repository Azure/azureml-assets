# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with File Search tool.

Creates a vector store, uploads a product catalog document,
then evaluates the agent's ability to answer questions about it.
No external connection IDs required beyond the project endpoint.
"""

import os
import logging
import pytest
from azure.ai.projects.models import PromptAgentDefinition, FileSearchTool

from conftest import run_evaluation, assert_evaluation_results, unique_name

logger = logging.getLogger(__name__)

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
PRODUCT_FILE = os.path.join(ASSET_DIR, "sample_product_info.txt")


@pytest.mark.flaky(reruns=3, reason="LLM quality scoring variability")
class TestFileSearchEvaluation:
    """Test agent evaluation with File Search tool."""

    def test_evaluate_agent_with_file_search(self, project_client, openai_client, model_deployment_name):
        """Evaluate an agent that uses File Search on a vector store."""
        # Setup: create vector store and upload document
        vector_store = openai_client.vector_stores.create(name=unique_name("E2E-VectorStore"))
        openai_client.vector_stores.files.upload_and_poll(
            vector_store_id=vector_store.id,
            file=open(PRODUCT_FILE, "rb"),
        )

        tool = FileSearchTool(vector_store_ids=[vector_store.id])
        agent_name = unique_name("E2E-FileSearch")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions="You are a helpful assistant. Search the uploaded files to answer product questions.",
                tools=[tool],
            ),
        )

        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}

            response = openai_client.responses.create(
                input="What is the price and battery life of the Contoso SmartWatch X100?",
                extra_body=extra_body,
            )
            assert response.output_text, "No text output from agent"

            logger.info("Response:\n%s", response.model_dump_json(indent=2))

            eval_run, output_items = run_evaluation(
                openai_client,
                model_deployment_name,
                response.id,
                agent.name,
                f"File Search E2E - {agent_name}",
            )
            assert_evaluation_results(eval_run, output_items)

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
            # Cleanup vector store
            openai_client.vector_stores.delete(vector_store_id=vector_store.id)
