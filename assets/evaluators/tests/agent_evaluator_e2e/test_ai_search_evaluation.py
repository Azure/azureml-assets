# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with Azure AI Search tool.

Requires:
    AI_SEARCH_PROJECT_CONNECTION_ID
    AI_SEARCH_INDEX_NAME
"""

import os
import logging
import pytest
from azure.ai.projects.models import (
    PromptAgentDefinition,
    AzureAISearchTool,
    AzureAISearchToolResource,
    AISearchIndexResource,
    AzureAISearchQueryType,
)

from conftest import (
    run_evaluation,
    assert_evaluation_results,
    unique_name,
    requires_env,
    UNSUPPORTED_TOOL_EVALUATORS,
)

logger = logging.getLogger(__name__)


@requires_env("AI_SEARCH_PROJECT_CONNECTION_ID", "AI_SEARCH_INDEX_NAME")
@pytest.mark.flaky(reruns=3, reason="LLM quality scoring variability")
class TestAISearchEvaluation:
    """Test agent evaluation with Azure AI Search tool."""

    def test_evaluate_agent_with_ai_search(self, project_client, openai_client, model_deployment_name):
        """Evaluate an agent that uses Azure AI Search."""
        tool = AzureAISearchTool(
            azure_ai_search=AzureAISearchToolResource(
                indexes=[
                    AISearchIndexResource(
                        project_connection_id=os.environ["AI_SEARCH_PROJECT_CONNECTION_ID"],
                        index_name=os.environ["AI_SEARCH_INDEX_NAME"],
                        query_type=AzureAISearchQueryType.SIMPLE,
                    ),
                ]
            )
        )
        agent_name = unique_name("E2E-AISearch")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions=(
                    "You are a helpful assistant. Use the search tool to find relevant information. "
                    "Cite sources when possible."
                ),
                tools=[tool],
            ),
        )

        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}

            response = openai_client.responses.create(
                input="Search for the most relevant information available in the index.",
                tool_choice="required",
                extra_body=extra_body,
            )
            assert response.output_text, "No text output from agent"

            logger.info("Response:\n%s", response.model_dump_json(indent=2))

            eval_run, output_items = run_evaluation(
                openai_client,
                model_deployment_name,
                response.id,
                agent.name,
                f"AI Search E2E - {agent_name}",
            )
            assert_evaluation_results(
                eval_run,
                output_items,
                expected_not_applicable=UNSUPPORTED_TOOL_EVALUATORS,
            )

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
