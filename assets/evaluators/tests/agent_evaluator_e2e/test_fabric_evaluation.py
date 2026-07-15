# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with Microsoft Fabric tool.

Requires:
    FABRIC_PROJECT_CONNECTION_ID
"""

import os
import logging
import pytest
from azure.ai.projects.models import (
    PromptAgentDefinition,
    MicrosoftFabricPreviewTool,
    FabricDataAgentToolParameters,
    ToolProjectConnection,
)

from conftest import (
    run_evaluation,
    assert_evaluation_results,
    unique_name,
    requires_env,
)

logger = logging.getLogger(__name__)


@requires_env("FABRIC_PROJECT_CONNECTION_ID")
@pytest.mark.flaky(reruns=3, reason="LLM quality scoring variability")
class TestFabricEvaluation:
    """Test agent evaluation with Microsoft Fabric tool."""

    def test_evaluate_agent_with_fabric(self, project_client, openai_client, eval_client, model_deployment_name):
        """Evaluate an agent that uses a Microsoft Fabric tool."""
        tool = MicrosoftFabricPreviewTool(
            fabric_dataagent_preview=FabricDataAgentToolParameters(
                project_connections=[
                    ToolProjectConnection(project_connection_id=os.environ["FABRIC_PROJECT_CONNECTION_ID"])
                ]
            )
        )
        agent_name = unique_name("E2E-Fabric")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions="You are a data analyst assistant. Use Fabric to query and analyze data.",
                tools=[tool],
            ),
        )

        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}
            user_input = os.environ.get(
                "FABRIC_USER_INPUT",
                "What data is available in the connected Fabric workspace?",
            )

            response = openai_client.responses.create(
                input=user_input,
                tool_choice="required",
                extra_body=extra_body,
            )
            assert response.output_text, "No text output from agent"

            logger.info("Response:\n%s", response.model_dump_json(indent=2))

            eval_run, output_items = run_evaluation(
                eval_client,
                model_deployment_name,
                response.id,
                agent.name,
                f"Fabric E2E - {agent_name}",
            )
            assert_evaluation_results(
                eval_run,
                output_items,
                # Only groundedness stays NOT_APPLICABLE for azure_fabric now.
                expected_not_applicable={"groundedness"},
                # Fabric may not find data / list workspace contents, so quality
                # and tool-call evaluators may penalize the "not found" response;
                # outcomes vary run to run.
                tolerated_failures={
                    "relevance",
                    "intent_resolution",
                    "task_completion",
                    "tool_call_success",
                    "tool_output_utilization",
                    "tool_call_accuracy",
                    "tool_input_accuracy",
                },
            )

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
