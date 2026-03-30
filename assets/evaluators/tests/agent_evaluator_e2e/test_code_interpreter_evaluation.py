# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with Code Interpreter tool.

The agent uses the built-in Code Interpreter to run code.
No external connections required.
"""

import logging
import pytest
from azure.ai.projects.models import PromptAgentDefinition, CodeInterpreterTool

from conftest import run_evaluation, assert_evaluation_results, unique_name, UNSUPPORTED_TOOL_EVALUATORS

logger = logging.getLogger(__name__)


@pytest.mark.flaky(reruns=3, reason="LLM quality scoring variability")
class TestCodeInterpreterEvaluation:
    """Test agent evaluation with Code Interpreter tool."""

    def test_evaluate_agent_with_code_interpreter(self, project_client, openai_client, model_deployment_name):
        """Evaluate an agent that uses the Code Interpreter."""
        tool = CodeInterpreterTool()
        agent_name = unique_name("E2E-CodeInterpreter")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions="You are a helpful math assistant. Use code interpreter to solve problems.",
                tools=[tool],
            ),
        )

        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}

            response = openai_client.responses.create(
                input="Calculate the first 10 prime numbers and return them as a list.",
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
                f"Code Interpreter E2E - {agent_name}",
            )
            assert_evaluation_results(
                eval_run,
                output_items,
                expected_not_applicable=UNSUPPORTED_TOOL_EVALUATORS,
            )

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
