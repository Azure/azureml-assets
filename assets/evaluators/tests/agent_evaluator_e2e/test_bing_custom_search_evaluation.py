# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with Bing Custom Search tool.

Requires:
    BING_CUSTOM_SEARCH_PROJECT_CONNECTION_ID
    BING_CUSTOM_SEARCH_INSTANCE_NAME
"""

import os
import logging
import pytest
from azure.ai.projects.models import (
    PromptAgentDefinition,
    BingCustomSearchPreviewTool,
    BingCustomSearchToolParameters,
    BingCustomSearchConfiguration,
)

from conftest import (
    run_evaluation,
    assert_evaluation_results,
    unique_name,
    requires_env,
    UNSUPPORTED_TOOL_EVALUATORS,
)

logger = logging.getLogger(__name__)


@requires_env("BING_CUSTOM_SEARCH_PROJECT_CONNECTION_ID", "BING_CUSTOM_SEARCH_INSTANCE_NAME")
@pytest.mark.flaky(reruns=3, reason="LLM quality scoring variability")
class TestBingCustomSearchEvaluation:
    """Test agent evaluation with Bing Custom Search tool."""

    def test_evaluate_agent_with_bing_custom_search(self, project_client, openai_client, model_deployment_name):
        """Evaluate an agent that uses Bing Custom Search."""
        tool = BingCustomSearchPreviewTool(
            bing_custom_search_preview=BingCustomSearchToolParameters(
                search_configurations=[
                    BingCustomSearchConfiguration(
                        project_connection_id=os.environ["BING_CUSTOM_SEARCH_PROJECT_CONNECTION_ID"],
                        instance_name=os.environ["BING_CUSTOM_SEARCH_INSTANCE_NAME"],
                    )
                ]
            )
        )
        agent_name = unique_name("E2E-BingCustomSearch")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions=(
                    "You are a helpful assistant. Use Bing Custom Search to find"
                    " relevant information from curated websites."
                ),
                tools=[tool],
            ),
        )

        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}

            stream_response = openai_client.responses.create(
                input="Search for the latest product announcements.",
                tool_choice="required",
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

            eval_run, output_items = run_evaluation(
                openai_client,
                model_deployment_name,
                response.id,
                agent.name,
                f"Bing Custom Search E2E - {agent_name}",
            )
            assert_evaluation_results(
                eval_run,
                output_items,
                expected_not_applicable=UNSUPPORTED_TOOL_EVALUATORS,
            )

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
