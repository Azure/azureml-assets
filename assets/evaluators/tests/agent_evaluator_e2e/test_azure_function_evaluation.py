# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with Azure Function tool.

NOTE: There are no unit tests for the Azure Function tool type in
base_tool_evaluation_test.py.  Failures in this test can be ignored
until unit-test coverage is added and expected behavior is established.

Requires:
    STORAGE_INPUT_QUEUE_NAME
    STORAGE_OUTPUT_QUEUE_NAME
    STORAGE_QUEUE_SERVICE_ENDPOINT
"""

import os
import logging
from azure.ai.projects.models import (
    PromptAgentDefinition,
    AzureFunctionTool,
    AzureFunctionDefinition,
    AzureFunctionBinding,
    AzureFunctionStorageQueue,
    AzureFunctionDefinitionFunction,
)

from conftest import (
    run_evaluation,
    assert_evaluation_results,
    unique_name,
    requires_env,
)

logger = logging.getLogger(__name__)


@requires_env(
    "STORAGE_INPUT_QUEUE_NAME",
    "STORAGE_OUTPUT_QUEUE_NAME",
    "STORAGE_QUEUE_SERVICE_ENDPOINT",
)
class TestAzureFunctionEvaluation:
    """Test agent evaluation with Azure Function tool."""

    def test_evaluate_agent_with_azure_function(self, project_client, openai_client, model_deployment_name):
        """Evaluate an agent that uses an Azure Function."""
        tool = AzureFunctionTool(
            azure_function=AzureFunctionDefinition(
                input_binding=AzureFunctionBinding(
                    storage_queue=AzureFunctionStorageQueue(
                        queue_name=os.environ["STORAGE_INPUT_QUEUE_NAME"],
                        queue_service_endpoint=os.environ["STORAGE_QUEUE_SERVICE_ENDPOINT"],
                    )
                ),
                output_binding=AzureFunctionBinding(
                    storage_queue=AzureFunctionStorageQueue(
                        queue_name=os.environ["STORAGE_OUTPUT_QUEUE_NAME"],
                        queue_service_endpoint=os.environ["STORAGE_QUEUE_SERVICE_ENDPOINT"],
                    )
                ),
                function=AzureFunctionDefinitionFunction(
                    name="queue_trigger",
                    description="Get weather for a given location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City or location to get weather for",
                            }
                        },
                    },
                ),
            )
        )
        agent_name = unique_name("E2E-AzureFunction")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions="You are a helpful assistant. Use the Azure Function tool to get weather information.",
                tools=[tool],
            ),
        )

        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}

            response = openai_client.responses.create(
                input="What is the weather in New York City?",
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
                f"Azure Function E2E - {agent_name}",
            )
            assert_evaluation_results(eval_run, output_items)

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
