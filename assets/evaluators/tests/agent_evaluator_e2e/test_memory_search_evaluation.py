# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with Memory Search tool.

Requires:
    MEMORY_STORE_CHAT_MODEL_DEPLOYMENT_NAME
    MEMORY_STORE_EMBEDDING_MODEL_DEPLOYMENT_NAME
    MEMORY_STORE_NAME (defaults to 'my_memory_store')

This test uses an existing memory store, seeds it with a conversation,
waits for memory extraction, then evaluates a follow-up conversation.
"""

import os
import time
import logging

import pytest
from azure.ai.projects.models import (
    PromptAgentDefinition,
    MemorySearchPreviewTool,
)

from conftest import (
    run_evaluation,
    assert_evaluation_results,
    unique_name,
    requires_env,
)

logger = logging.getLogger(__name__)


@requires_env(
    "MEMORY_STORE_CHAT_MODEL_DEPLOYMENT_NAME",
    "MEMORY_STORE_EMBEDDING_MODEL_DEPLOYMENT_NAME",
)
@pytest.mark.flaky(reruns=3, reason="LLM quality scoring variability")
class TestMemorySearchEvaluation:
    """Test agent evaluation with Memory Search tool."""

    def test_evaluate_agent_with_memory_search(self, project_client, openai_client, model_deployment_name):
        """Evaluate an agent that uses a memory store for search."""
        memory_store_name = os.environ.get("MEMORY_STORE_NAME", "my_memory_store")
        user_scope = unique_name("e2e_user")

        tool = MemorySearchPreviewTool(
            memory_store_name=memory_store_name,
            scope=user_scope,
            update_delay=1,
        )
        agent_name = unique_name("E2E-MemorySearch")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=model_deployment_name,
                instructions="You are a helpful assistant with memory. Remember what the user tells you.",
                tools=[tool],
            ),
        )

        conv1 = None
        conv2 = None
        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}

            # First conversation – seed memory
            conv1 = openai_client.conversations.create(
                items=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": "My favorite color is blue and I live in Portland, Oregon.",
                    }
                ],
            )
            openai_client.responses.create(
                conversation=conv1.id,
                extra_body=extra_body,
            )

            # Wait for memory extraction
            time.sleep(120)

            # Second conversation – test recall
            conv2 = openai_client.conversations.create(
                items=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": "What is my favorite color and where do I live?",
                    }
                ],
            )
            response = openai_client.responses.create(
                conversation=conv2.id,
                extra_body=extra_body,
            )
            assert response.output_text, "No text output from agent"

            logger.info("Response:\n%s", response.model_dump_json(indent=2))

            eval_run, output_items = run_evaluation(
                openai_client,
                model_deployment_name,
                response.id,
                agent.name,
                f"Memory Search E2E - {agent_name}",
            )
            assert_evaluation_results(
                eval_run,
                output_items,
                # Evaluator can't see the memory store context from
                # the first conversation, so it views recall as fabrication.

            )

        finally:
            # Cleanup
            for conv in (conv1, conv2):
                if conv:
                    try:
                        openai_client.conversations.delete(conversation_id=conv.id)
                    except Exception:
                        pass
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
