# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with KB MCP (Knowledge Base) tool.

Uses an MCP server backed by a knowledge base.  The tool type in unit tests
is ``function`` (not in UNSUPPORTED_TOOLS), so all evaluators should pass.

Requires:
    MCP_KB_SERVER_URL
    MCP_KB_PROJECT_CONNECTION_ID
"""

import os
import logging
import pytest
from azure.ai.projects.models import PromptAgentDefinition, MCPTool

from conftest import (
    run_evaluation,
    assert_evaluation_results,
    unique_name,
    requires_env,
)

logger = logging.getLogger(__name__)


@requires_env("MCP_KB_SERVER_URL", "MCP_KB_PROJECT_CONNECTION_ID")
@pytest.mark.flaky(reruns=3, reason="LLM quality scoring variability")
class TestKBMCPEvaluation:
    """Test agent evaluation with Knowledge Base MCP tool."""

    def test_evaluate_agent_with_kb_mcp(self, project_client, openai_client, model_deployment_name):
        """Evaluate an agent that uses a Knowledge Base via MCP."""
        server_url = os.environ["MCP_KB_SERVER_URL"]
        connection_id = os.environ["MCP_KB_PROJECT_CONNECTION_ID"]

        tool = MCPTool(
            server_label="KnowledgeBase",
            server_url=server_url,
            require_approval={"never": {"tool_names": ["knowledge_base_retrieve"]}},
            project_connection_id=connection_id,
        )
        agent_name = unique_name("E2E-KBMCP")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions="You are a helpful assistant that uses the knowledge-base retrieve tool to answer questions about earth and geology. You MUST call the tool.",
                tools=[tool],
            ),
        )

        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}

            response = openai_client.responses.create(
                input="What's the size of Africa?",
                extra_body=extra_body,
            )
            assert response.output_text, "No text output from agent"

            logger.info("Response:\n%s", response.model_dump_json(indent=2))

            eval_run, output_items = run_evaluation(
                openai_client,
                model_deployment_name,
                response.id,
                agent.name,
                f"KB MCP E2E - {agent_name}",
            )
            assert_evaluation_results(
                eval_run,
                output_items,
                expected_failures={"tool_output_utilization", "tool_call_success"},
                tolerated_failures={"groundedness", "task_adherence"},
            )

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
