# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with Web Search tool.

No external connection IDs required – the tool uses the built-in web search.
Uses streaming to consume the response and avoid content filter issues.
"""

import logging
import pytest
from azure.ai.projects.models import (
    PromptAgentDefinition,
    WebSearchPreviewTool,
    WebSearchApproximateLocation,
)

from conftest import (
    unique_name,
    run_evaluation,
    assert_evaluation_results,
    UNSUPPORTED_TOOL_EVALUATORS,
)

logger = logging.getLogger(__name__)


@pytest.mark.flaky(reruns=3, reason="LLM quality scoring variability")
class TestWebSearchEvaluation:
    """Test agent evaluation with Web Search tool."""

    def test_evaluate_agent_with_web_search(self, project_client, openai_client, model_deployment_name):
        """Evaluate an agent that uses the Web Search tool."""
        tool = WebSearchPreviewTool(
            user_location=WebSearchApproximateLocation(country="US", city="Seattle", region="Washington")
        )
        agent_name = unique_name("E2E-WebSearch")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions=(
                    "You are a helpful assistant. Use web search to answer"
                    " questions with current information."
                ),
                tools=[tool],
            ),
        )

        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}

            stream_response = openai_client.responses.create(
                input="What is the current population of Seattle, Washington?",
                tool_choice="auto",
                stream=True,
                store=True,
                extra_body=extra_body,
            )

            response = None
            for event in stream_response:
                logger.info("Stream event: %s", event.type)
                if event.type == "response.created":
                    logger.info("Response created with ID: %s", event.response.id)
                elif event.type == "response.output_text.delta":
                    logger.info("Delta: %s", event.delta)
                elif event.type == "response.text.done":
                    logger.info("Response text done!")
                elif event.type == "response.output_item.done":
                    if event.item.type == "message":
                        item = event.item
                        if item.content[-1].type == "output_text":
                            text_content = item.content[-1]
                            for annotation in text_content.annotations:
                                if annotation.type == "url_citation":
                                    logger.info(
                                        "URL Citation: %s, Start index: %s, End index: %s",
                                        annotation.url,
                                        annotation.start_index,
                                        annotation.end_index,
                                    )
                elif event.type == "response.completed":
                    response = event.response
                    logger.info("Response completed!")
                    logger.info("Full response: %s", event.response.output_text)
                elif event.type == "response.incomplete":
                    response = event.response
                    logger.warning("Response incomplete: %s", getattr(event.response, 'incomplete_details', None))

            assert response is not None, "Never received response.completed or response.incomplete event"
            logger.info("Response:\n%s", response.model_dump_json(indent=2))
            assert response.output_text, "No text output from agent"

            output_types = [item.type for item in response.output]
            logger.info("Web search response ID: %s", response.id)
            logger.info("Web search output types: %s", output_types)

            # Verify web search was actually used
            assert any(
                t == "web_search_call" for t in output_types
            ), "Expected web_search_call in output but got: " + str(output_types)

            eval_run, output_items = run_evaluation(
                openai_client,
                model_deployment_name,
                response.id,
                agent.name,
                f"Web Search E2E - {agent_name}",
            )
            assert_evaluation_results(
                eval_run,
                output_items,
                expected_not_applicable=UNSUPPORTED_TOOL_EVALUATORS,
                expected_failures={"task_adherence"},
            )

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
