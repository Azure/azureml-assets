# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with SharePoint tool.

Requires:
    SHAREPOINT_PROJECT_CONNECTION_ID
"""

import os
import logging
import pytest
from azure.ai.projects.models import (
    PromptAgentDefinition,
    SharepointPreviewTool,
    SharepointGroundingToolParameters,
    ToolProjectConnection,
)

from conftest import (
    run_evaluation,
    assert_evaluation_results,
    unique_name,
    requires_env,
)

logger = logging.getLogger(__name__)


@requires_env("SHAREPOINT_PROJECT_CONNECTION_ID")
@pytest.mark.flaky(reruns=3, reason="LLM quality scoring variability")
class TestSharePointEvaluation:
    """Test agent evaluation with SharePoint tool."""

    def test_evaluate_agent_with_sharepoint(self, project_client, openai_client, eval_client, model_deployment_name):
        """Evaluate an agent that uses a SharePoint tool."""
        tool = SharepointPreviewTool(
            sharepoint_grounding_preview=SharepointGroundingToolParameters(
                project_connections=[
                    ToolProjectConnection(project_connection_id=os.environ["SHAREPOINT_PROJECT_CONNECTION_ID"])
                ]
            )
        )
        agent_name = unique_name("E2E-SharePoint")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions="You are a helpful assistant. Search SharePoint for relevant documents.",
                tools=[tool],
            ),
        )

        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}
            user_input = os.environ.get(
                "SHAREPOINT_USER_INPUT",
                "Find information about company policies.",
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
                f"SharePoint E2E - {agent_name}",
            )
            assert_evaluation_results(
                eval_run,
                output_items,
                # Only groundedness stays NOT_APPLICABLE for sharepoint_grounding now.
                expected_not_applicable={"groundedness"},
                # SharePoint may not find matching documents, so quality and
                # tool-call evaluators may penalize the "not found" response;
                # outcomes vary run to run.
                tolerated_failures={
                    "relevance",
                    "intent_resolution",
                    "task_completion",
                    "tool_call_accuracy",
                    "tool_input_accuracy",
                },
            )

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
