# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with Browser Automation tool.

Requires:
    BROWSER_AUTOMATION_PROJECT_CONNECTION_ID
"""

import os
import logging
import pytest
from azure.ai.projects.models import (
    PromptAgentDefinition,
    BrowserAutomationPreviewTool,
    BrowserAutomationToolParameters,
    BrowserAutomationToolConnectionParameters,
)

from conftest import (
    run_evaluation,
    assert_evaluation_results,
    unique_name,
    requires_env,
    UNSUPPORTED_TOOL_EVALUATORS,
)

logger = logging.getLogger(__name__)


@requires_env("BROWSER_AUTOMATION_PROJECT_CONNECTION_ID")
@pytest.mark.flaky(reruns=3, reason="LLM quality scoring variability")
class TestBrowserAutomationEvaluation:
    """Test agent evaluation with Browser Automation tool."""

    def test_evaluate_agent_with_browser_automation(self, project_client, openai_client, model_deployment_name):
        """Evaluate an agent that uses Browser Automation."""
        tool = BrowserAutomationPreviewTool(
            browser_automation_preview=BrowserAutomationToolParameters(
                connection=BrowserAutomationToolConnectionParameters(
                    project_connection_id=os.environ["BROWSER_AUTOMATION_PROJECT_CONNECTION_ID"],
                )
            )
        )
        agent_name = unique_name("E2E-BrowserAutomation")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions="You are a helpful assistant that can automate browser tasks.",
                tools=[tool],
            ),
        )

        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}

            response = openai_client.responses.create(
                input="Navigate to https://example.com and tell me what the page title says.",
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
                f"Browser Automation E2E - {agent_name}",
            )
            assert_evaluation_results(
                eval_run,
                output_items,
                expected_not_applicable=UNSUPPORTED_TOOL_EVALUATORS,
            )

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
