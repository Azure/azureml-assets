# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""E2E evaluation test for an agent with Computer Use tool.

Requires:
    COMPUTER_USE_MODEL_DEPLOYMENT_NAME (defaults to 'computer-use-preview')

Based on the official SDK sample:
https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ai/azure-ai-projects/samples/agents/tools/sample_agent_computer_use.py

The test creates a Computer Use Agent that performs a simulated web search,
feeding it pre-recorded screenshots to drive a deterministic state machine
(browser_search → typed → search_results).
"""

import base64
import logging
import os
import pytest
from enum import Enum
from pathlib import Path

from azure.ai.projects.models import ComputerUsePreviewTool, PromptAgentDefinition

from conftest import (
    assert_evaluation_results,
    requires_env,
    run_evaluation,
    unique_name,
)

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).parent / "assets"

# ---------------------------------------------------------------------------
# Computer-use helpers (mirrors computer_use_util.py from Azure SDK samples)
# ---------------------------------------------------------------------------

class SearchState(Enum):
    INITIAL = "initial"
    TYPED = "typed"
    PRESSED_ENTER = "pressed_enter"


def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _load_screenshot_assets() -> dict:
    mapping = {
        "browser_search": "cua_browser_search.png",
        "search_typed": "cua_search_typed.png",
        "search_results": "cua_search_results.png",
    }
    screenshots = {}
    for key, filename in mapping.items():
        path = ASSETS_DIR / filename
        b64 = _image_to_base64(str(path))
        screenshots[key] = {
            "filename": filename,
            "url": f"data:image/png;base64,{b64}",
        }
    return screenshots


def _handle_action_and_screenshot(action, current_state, screenshots):
    """Process a computer action and return the next screenshot + state."""
    logger.info("Executing computer action: %s", action.type)

    if action.type == "type" and getattr(action, "text", None):
        current_state = SearchState.TYPED
        logger.info("  Typing text: '%s'", action.text)
    elif (
        action.type in ("key", "keypress")
        and getattr(action, "keys", None)
        and ("Return" in str(action.keys) or "ENTER" in str(action.keys))
    ):
        current_state = SearchState.PRESSED_ENTER
        logger.info("  Detected ENTER key press")
    elif action.type == "click" and current_state == SearchState.TYPED:
        current_state = SearchState.PRESSED_ENTER
        logger.info("  Detected click after typing (submit)")

    if current_state == SearchState.PRESSED_ENTER:
        screenshot_info = screenshots["search_results"]
    elif current_state == SearchState.TYPED:
        screenshot_info = screenshots["search_typed"]
    else:
        screenshot_info = screenshots["browser_search"]

    return screenshot_info, current_state


@requires_env("COMPUTER_USE_MODEL_DEPLOYMENT_NAME")
@pytest.mark.flaky(reruns=3, reason="LLM quality scoring variability")
class TestComputerUseEvaluation:
    """Test agent evaluation with Computer Use tool."""

    def test_evaluate_agent_with_computer_use(self, project_client, openai_client, model_deployment_name):
        """Evaluate an agent that uses the Computer Use tool."""
        computer_model = os.environ.get("COMPUTER_USE_MODEL_DEPLOYMENT_NAME", "computer-use-preview")

        screenshots = _load_screenshot_assets()

        tool = ComputerUsePreviewTool(display_width=1026, display_height=769, environment="windows")
        agent_name = unique_name("E2E-ComputerUse")
        agent = project_client.agents.create_version(
            agent_name=agent_name,
            definition=PromptAgentDefinition(
                model=computer_model,
                instructions=(
                    "You are a computer automation assistant. "
                    "Be direct and efficient. When you reach the search results page, "
                    "read and describe the actual search result titles and descriptions you can see."
                ),
                tools=[tool],
            ),
            description="Computer automation agent with screen interaction capabilities.",
        )

        try:
            extra_body = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}

            # Initial request with screenshot — start with Bing search page
            response = openai_client.responses.create(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "I need you to help me search for 'OpenAI news'. "
                                    "Please type 'OpenAI news' and submit the search. "
                                    "Once you see search results, the task is complete."
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": screenshots["browser_search"]["url"],
                                "detail": "high",
                            },
                        ],
                    }
                ],
                extra_body=extra_body,
                truncation="auto",
            )

            logger.info("Initial response received (ID: %s)", response.id)
            logger.info("Initial response:\n%s", response.model_dump_json(indent=2))

            # Main interaction loop with deterministic completion
            current_state = SearchState.INITIAL
            max_iterations = 10
            iteration = 0

            while True:
                if iteration >= max_iterations:
                    logger.warning("Reached maximum iterations (%d). Stopping.", max_iterations)
                    break
                iteration += 1
                logger.info("--- Iteration %d ---", iteration)

                computer_calls = [item for item in response.output if item.type == "computer_call"]

                if not computer_calls:
                    # No more computer calls — agent completed the task
                    for item in response.output:
                        if item.type == "message":
                            for part in item.content:
                                text = getattr(part, "text", None)
                                if text:
                                    logger.info("Agent final output: %s", text)
                    break

                computer_call = computer_calls[0]
                action = computer_call.action
                call_id = computer_call.call_id

                logger.info("Processing computer call (ID: %s, action: %s)", call_id, action.type)

                screenshot_info, current_state = _handle_action_and_screenshot(
                    action, current_state, screenshots
                )

                logger.info("Sending action result (screenshot: %s)", screenshot_info["filename"])

                response = openai_client.responses.create(
                    previous_response_id=response.id,
                    input=[
                        {
                            "call_id": call_id,
                            "type": "computer_call_output",
                            "output": {
                                "type": "computer_screenshot",
                                "image_url": screenshot_info["url"],
                            },
                        }
                    ],
                    extra_body=extra_body,
                    truncation="auto",
                )

                logger.info("Follow-up response received (ID: %s)", response.id)
                logger.info("Follow-up response:\n%s", response.model_dump_json(indent=2))

            logger.info("Final response:\n%s", response.model_dump_json(indent=2))

            eval_run, output_items = run_evaluation(
                openai_client,
                model_deployment_name,
                response.id,
                agent.name,
                f"Computer Use E2E - {agent_name}",
            )
            assert_evaluation_results(
                eval_run,
                output_items,
                # computer_call uses non-standard arguments format causing
                # "The 'arguments' field must be a dictionary" errors.
                # See known_issues.md §1.
                expected_errors={
                    "coherence": "arguments' field must be a dictionary",
                    "fluency": "arguments' field must be a dictionary",
                    "groundedness": "arguments' field must be a dictionary",
                    "relevance": "arguments' field must be a dictionary",
                    "intent_resolution": "arguments' field must be a dictionary",
                    "task_adherence": "arguments' field must be a dictionary",
                    "task_completion": "arguments' field must be a dictionary",
                    "tool_call_success": "arguments' field must be a dictionary",
                    "tool_call_accuracy": "Tool definitions input is required but not provided",
                    "tool_selection": "Tool definitions input is required but not provided",
                    "tool_input_accuracy": "arguments' field must be a dictionary",
                    "tool_output_utilization": "arguments' field must be a dictionary",
                },
            )

        finally:
            project_client.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
