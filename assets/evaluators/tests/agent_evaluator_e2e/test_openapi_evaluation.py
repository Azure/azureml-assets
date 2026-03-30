# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with OpenAPI tool.

Uses the public wttr.in weather API (no auth required).
The OpenAPI spec is loaded from the assets folder.
"""

import json
import logging
import pytest
from pathlib import Path

from azure.ai.projects.models import (
    PromptAgentDefinition,
    OpenApiTool,
    OpenApiFunctionDefinition,
    OpenApiAnonymousAuthDetails,
)

from conftest import run_evaluation, assert_evaluation_results, unique_name, UNSUPPORTED_TOOL_EVALUATORS

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).parent / "assets"

with open(ASSETS_DIR / "weather_openapi.json") as f:
    WEATHER_OPENAPI_SPEC = json.load(f)


@pytest.mark.flaky(reruns=3, reason="LLM quality scoring variability")
class TestOpenAPIEvaluation:
    """Test agent evaluation with OpenAPI tool."""

    def test_evaluate_agent_with_openapi(self, project_client, openai_client, model_deployment_name):
        """Evaluate an agent that calls an OpenAPI endpoint."""
        tool = OpenApiTool(
            openapi=OpenApiFunctionDefinition(
                name="get_weather",
                spec=WEATHER_OPENAPI_SPEC,
                description="Retrieve weather information for a location.",
                auth=OpenApiAnonymousAuthDetails(),
            )
        )
        agent_name = unique_name("E2E-OpenAPI")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions="You are a helpful assistant that retrieves weather information using the API.",
                tools=[tool],
            ),
        )

        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}

            response = openai_client.responses.create(
                input="What's the weather like in Cairo?",
                extra_body=extra_body,
            )
            assert response.output_text, "No text output from agent"

            logger.info("Response:\n%s", response.model_dump_json(indent=2))

            eval_run, output_items = run_evaluation(
                openai_client,
                model_deployment_name,
                response.id,
                agent.name,
                f"OpenAPI E2E - {agent_name}",
            )
            assert_evaluation_results(
                eval_run,
                output_items,
                # openapi_call is in UNSUPPORTED_TOOLS – 4 evaluators
                # return NOT_APPLICABLE per base_tool_evaluation_test.py.
                # tool_call_accuracy and tool_selection also check unsupported
                # tools BUT have is_tool_definition_required=True; the
                # missing-definitions check fires before the unsupported check.
                expected_not_applicable=UNSUPPORTED_TOOL_EVALUATORS - {"tool_call_accuracy"},
                expected_errors={
                    "tool_call_accuracy": "Tool definitions input is required but not provided",
                    "tool_selection": "Tool definitions input is required but not provided",
                },
                # openapi_call tool type is not recognized as tool usage by
                # task_adherence evaluator (known platform limitation)
                expected_failures={"task_adherence"},
            )

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
