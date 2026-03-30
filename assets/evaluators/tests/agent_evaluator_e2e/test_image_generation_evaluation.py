# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with Image Generation tool.

Requires:
    IMAGE_GENERATION_MODEL_DEPLOYMENT_NAME
"""

import os
import logging
import pytest
from azure.ai.projects.models import PromptAgentDefinition, ImageGenTool

from conftest import (
    run_evaluation,
    assert_evaluation_results,
    unique_name,
    requires_env,
)

logger = logging.getLogger(__name__)


@requires_env("IMAGE_GENERATION_MODEL_DEPLOYMENT_NAME")
@pytest.mark.flaky(reruns=3, reason="LLM quality scoring variability")
class TestImageGenerationEvaluation:
    """Test agent evaluation with Image Generation tool."""

    def test_evaluate_agent_with_image_generation(self, project_client, openai_client, model_deployment_name):
        """Evaluate an agent that uses the Image Generation tool."""
        image_model = os.environ["IMAGE_GENERATION_MODEL_DEPLOYMENT_NAME"]
        tool = ImageGenTool(
            model=image_model,
            quality="low",
            size="1024x1024",
        )
        agent_name = unique_name("E2E-ImageGen")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions="You are a creative assistant that generates images based on descriptions.",
                tools=[tool],
            ),
        )

        try:
            extra_body = {
                "agent_reference": {"name": agent.name, "type": "agent_reference"},
            }
            extra_headers = {
                "x-ms-oai-image-generation-deployment": image_model,
            }

            response = openai_client.responses.create(
                input="Generate an image of a sunset over a mountain lake.",
                extra_body=extra_body,
                extra_headers=extra_headers,
            )
            assert response.output_text, "No text output from agent"

            logger.info("Response:\n%s", response.model_dump_json(indent=2))

            eval_run, output_items = run_evaluation(
                openai_client,
                model_deployment_name,
                response.id,
                agent.name,
                f"Image Generation E2E - {agent_name}",
            )
            assert_evaluation_results(
                eval_run,
                output_items,
                # Evaluators can't view generated images, so they report
                # "no image delivered" for intent/task evaluators.
                expected_failures={
                    "intent_resolution",
                    "tool_input_accuracy",
                },
                tolerated_failures={
                    "task_completion",
                },
            )

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
