# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with a Function Tool.

The agent uses a locally-executed function tool (get_horoscope).
The test creates the agent, handles the function-call round-trip,
then evaluates the final response with all 13 quality evaluators.
"""

import json
import logging
import pytest
from azure.ai.projects.models import PromptAgentDefinition, FunctionTool, Tool
from openai.types.responses.response_input_param import FunctionCallOutput, ResponseInputParam

from conftest import run_evaluation, assert_evaluation_results, unique_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local function that the agent can call
# ---------------------------------------------------------------------------
def get_horoscope(sign: str) -> str:
    """Return a mock horoscope reading for the given zodiac sign."""
    return (
        f"{sign}: Today brings a wave of creativity and clarity. "
        f"You may find new opportunities emerging in your career or personal projects. "
        f"Stay open to unexpected conversations — they could lead to meaningful connections. "
        f"Lucky number: 7. Lucky color: turquoise."
    )


FUNCTION_TOOL = FunctionTool(
    name="get_horoscope",
    parameters={
        "type": "object",
        "properties": {
            "sign": {
                "type": "string",
                "description": "An astrological sign like Taurus or Aquarius",
            },
        },
        "required": ["sign"],
        "additionalProperties": False,
    },
    description="Get today's horoscope for an astrological sign.",
    strict=True,
)

TOOLS: list[Tool] = [FUNCTION_TOOL]


@pytest.mark.flaky(reruns=3, reason="LLM quality scoring variability")
class TestFunctionToolEvaluation:
    """Test agent evaluation with a local Function Tool."""

    def test_evaluate_agent_with_function_tool(self, project_client, openai_client, model_deployment_name):
        """Evaluate an agent that calls a local function tool."""
        agent_name = unique_name("E2E-FunctionTool")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions="You are a helpful assistant that can use function tools.",
                tools=TOOLS,
            ),
        )

        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}

            # First call – model should invoke get_horoscope
            response = openai_client.responses.create(
                input="What is my horoscope? I am an Aquarius.",
                extra_body=extra_body,
            )

            # Process function calls and send results back
            input_list: ResponseInputParam = []
            for item in response.output:
                if item.type == "function_call" and item.name == "get_horoscope":
                    result = get_horoscope(**json.loads(item.arguments))
                    input_list.append(
                        FunctionCallOutput(
                            type="function_call_output",
                            call_id=item.call_id,
                            output=json.dumps({"horoscope": result}),
                        )
                    )

            assert input_list, "Model did not produce any function calls"

            # Second call – model produces final text answer
            response = openai_client.responses.create(
                input=input_list,
                previous_response_id=response.id,
                extra_body=extra_body,
            )
            assert response.output_text, "No text output from agent"

            logger.info("Response:\n%s", response.model_dump_json(indent=2))

            # Evaluate
            eval_run, output_items = run_evaluation(
                openai_client,
                model_deployment_name,
                response.id,
                agent.name,
                f"Function Tool E2E - {agent_name}",
            )
            assert_evaluation_results(eval_run, output_items)

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
