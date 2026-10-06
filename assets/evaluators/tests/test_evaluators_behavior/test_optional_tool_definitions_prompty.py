# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Render the real tool evaluator Prompty files with and without ``tool_definitions``.

The behavior tests mock the judge flow, so they never load a Prompty. A legacy Prompty input that has no
``default`` raises ``MissingRequiredInputError`` when it is absent, so these tests are what guarantee that an
omitted ``tool_definitions`` input reaches the judge instead of failing before the model call.
"""

from pathlib import Path

import pytest
from azure.ai.evaluation._legacy.prompty._prompty import AsyncPrompty

BUILTIN_DIR = Path(__file__).resolve().parents[2] / "builtin"
MODEL = {
    "configuration": {"azure_endpoint": "https://example.openai.azure.com", "azure_deployment": "d", "api_key": "k"}
}
SENTINEL = "SENTINEL_TOOL_DEFINITION_PAYLOAD"
BASE_INPUTS = {"query": "q", "response": "r", "tool_calls": [{"name": "f"}]}

# (evaluator, text that precedes the rendered tool definitions in the Prompty body)
PROMPTY_CASES = [
    ("tool_call_success", "TOOL_DEFINITIONS: "),
    ("tool_call_accuracy", "TOOL DEFINITIONS: "),
    ("tool_selection", "TOOL DEFINITIONS: "),
    ("tool_input_accuracy", "## Tool Definitions:\n"),
    ("tool_output_utilization", "TOOL_DEFINITIONS: "),
]


def _render(name: str, **inputs) -> str:
    prompty = AsyncPrompty.load(source=BUILTIN_DIR / name / "evaluator" / f"{name}.prompty", model=MODEL)
    messages = prompty.render(**BASE_INPUTS, **inputs)
    return "\n".join(str(message.get("content")) for message in messages).replace("\r\n", "\n")


@pytest.mark.unittest
@pytest.mark.parametrize("name,prefix", PROMPTY_CASES)
class TestToolPromptyOptionalToolDefinitions:
    """tool_definitions is optional in every tool evaluator Prompty."""

    def test_definitions_are_rendered_when_provided(self, name, prefix):
        rendered = _render(name, tool_definitions=SENTINEL)
        assert prefix + SENTINEL in rendered

    @pytest.mark.parametrize(
        "absent", [{}, {"tool_definitions": None}, {"tool_definitions": ""}, {"tool_definitions": []}]
    )
    def test_absent_definitions_render_without_error(self, name, prefix, absent):
        rendered = _render(name, **absent)
        assert SENTINEL not in rendered
        assert rendered.strip()

    def test_absent_definitions_drop_the_definitions_block(self, name, prefix):
        with_definitions = _render(name, tool_definitions=SENTINEL)
        without_definitions = _render(name)
        assert with_definitions.count(prefix) == without_definitions.count(prefix) + 1
