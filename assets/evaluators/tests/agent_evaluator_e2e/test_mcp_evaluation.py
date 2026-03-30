# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with MCP (Model Context Protocol) tool.

Uses the Microsoft Learn MCP server.
"""

import os
import logging
import pytest
from azure.ai.projects.models import PromptAgentDefinition, MCPTool

from conftest import run_evaluation, assert_evaluation_results, unique_name

logger = logging.getLogger(__name__)


@pytest.mark.flaky(reruns=3, reason="LLM quality scoring variability")
class TestMCPEvaluation:
    """Test agent evaluation with MCP tool."""

    def test_evaluate_agent_with_mcp(self, project_client, openai_client, model_deployment_name):
        """Evaluate an agent that uses an MCP server."""
        server_url = os.environ.get("MCP_LEARN_SERVER_URL", "https://learn.microsoft.com/api/mcp")
        connection_id = os.environ.get("MCP_LEARN_PROJECT_CONNECTION_ID", "MicrosoftLearn2")

        tool = MCPTool(
            server_label="MicrosoftLearn2",
            server_url=server_url,
            require_approval={"never": {"tool_names": ["microsoft_docs_search", "microsoft_docs_fetch"]}},
            project_connection_id=connection_id,
        )
        agent_name = unique_name("E2E-MCP")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions="You are a helpful assistant that can search Microsoft Learn documentation. Use the MCP tool to look up Azure documentation. When you find relevant pages, fetch the full content for completeness.",
                tools=[tool],
            ),
        )

        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}

            response = openai_client.responses.create(
                input="can you tell me more about how azure functions work? use MCP",
                extra_body=extra_body,
            )
            assert response.output_text, "No text output from agent"

            logger.info("Response:\n%s", response.model_dump_json(indent=2))

            eval_run, output_items = run_evaluation(
                openai_client,
                model_deployment_name,
                response.id,
                agent.name,
                f"MCP E2E - {agent_name}",
            )
            assert_evaluation_results(
                eval_run,
                output_items,
            )

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
