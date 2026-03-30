# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with Agent-to-Agent (A2A) tool.

NOTE: There are no unit tests for the Agent-to-Agent tool type in
base_tool_evaluation_test.py.  Failures in this test can be ignored
until unit-test coverage is added and expected behavior is established.

Requires:
    A2A_PROJECT_CONNECTION_ID
"""

import os
import logging
from azure.ai.projects.models import PromptAgentDefinition, A2APreviewTool

from conftest import (
    run_evaluation,
    assert_evaluation_results,
    unique_name,
    requires_env,
)

logger = logging.getLogger(__name__)


@requires_env("A2A_PROJECT_CONNECTION_ID")
class TestAgentToAgentEvaluation:
    """Test agent evaluation with Agent-to-Agent tool."""

    def test_evaluate_agent_with_a2a(self, project_client, openai_client, model_deployment_name):
        """Evaluate an agent that uses the A2A tool."""
        tool = A2APreviewTool(
            project_connection_id=os.environ["A2A_PROJECT_CONNECTION_ID"],
        )
        # Optional: set base_url if the connection is missing a target
        if os.environ.get("A2A_ENDPOINT"):
            tool.base_url = os.environ["A2A_ENDPOINT"]

        agent_name = unique_name("E2E-A2A")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions="You are a coordinator agent. Delegate tasks to connected agents when appropriate.",
                tools=[tool],
            ),
        )

        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}
            user_input = os.environ.get(
                "A2A_USER_INPUT",
                "What capabilities do you have through your connected agents?",
            )

            response = openai_client.responses.create(
                input=user_input,
                extra_body=extra_body,
            )
            assert response.output_text, "No text output from agent"

            logger.info("Response:\n%s", response.model_dump_json(indent=2))

            eval_run, output_items = run_evaluation(
                openai_client,
                model_deployment_name,
                response.id,
                agent.name,
                f"Agent-to-Agent E2E - {agent_name}",
            )
            assert_evaluation_results(eval_run, output_items)

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
