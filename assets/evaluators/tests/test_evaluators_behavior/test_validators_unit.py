# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Direct unit tests for the evaluator input validators.

Validators are currently defined inline in each evaluator (duplicated, identical copies).
They are imported here from representative evaluators that define each class:
- ConversationValidator / ToolDefinitionsValidator / MessagesOrQueryResponseInputValidator
  + ValidatorInterface / MessageRole / ContentType / EvaluationLevel -> task_completion
- ToolCallsValidator -> tool_call_accuracy
- TaskNavigationEfficiencyValidator -> task_navigation_efficiency

When the shared-module refactor lands, only these import lines need to change.
"""

import pytest

from azure.ai.evaluation._exceptions import EvaluationException, ErrorCategory, ErrorTarget

from ...builtin.task_completion.evaluator._task_completion import (
    ConversationValidator,
    ContentType,
    MessageRole,
    MessagesOrQueryResponseInputValidator,
    ToolDefinitionsValidator,
)
from ...builtin.tool_call_accuracy.evaluator._tool_call_accuracy import ToolCallsValidator
from ...builtin.task_navigation_efficiency.evaluator._task_navigation_efficiency import (
    TaskNavigationEfficiencyValidator,
)

ET = ErrorTarget.GROUNDEDNESS_EVALUATOR

VALID_QUERY = [{"role": "user", "content": [{"type": "input_text", "text": "What is the weather?"}]}]
VALID_RESPONSE = [{"role": "assistant", "content": [{"type": "text", "text": "It is sunny."}]}]
VALID_TOOL_DEFINITIONS = [{"name": "get_weather", "parameters": {"type": "object", "properties": {}}}]
VALID_TOOL_CALLS = [{"type": "tool_call", "tool_call_id": "c1", "name": "get_weather", "arguments": {"city": "X"}}]


def _category(exc: EvaluationException) -> str:
    """Return the ErrorCategory name of an EvaluationException."""
    return exc.value.category.name if hasattr(exc, "value") else exc.category.name


@pytest.mark.unittest
class TestValidatorInterfaceContract:
    """Validators implement the ValidatorInterface contract."""

    @pytest.mark.parametrize(
        "cls",
        [ConversationValidator, ToolDefinitionsValidator, ToolCallsValidator,
         MessagesOrQueryResponseInputValidator, TaskNavigationEfficiencyValidator],
    )
    def test_implements_validator_interface(self, cls):
        """Every validator derives from a ValidatorInterface and exposes validate_eval_input.

        Note: validators are duplicated per-evaluator, so each module has its own
        ``ValidatorInterface`` class object. We therefore match by name across the MRO
        rather than by identity.
        """
        mro_names = {base.__name__ for base in cls.__mro__}
        assert "ValidatorInterface" in mro_names
        assert callable(getattr(cls, "validate_eval_input", None))



@pytest.mark.unittest
class TestConversationValidator:
    """Unit tests for ConversationValidator."""

    def test_valid_query_response_passes(self):
        """A well-formed query/response passes."""
        v = ConversationValidator(error_target=ET, requires_query=True)
        assert v.validate_eval_input({"query": VALID_QUERY, "response": VALID_RESPONSE}) is True

    def test_valid_conversation_passes(self):
        """A well-formed conversation dict passes."""
        v = ConversationValidator(error_target=ET)
        conversation = {"messages": VALID_QUERY + VALID_RESPONSE}
        assert v.validate_eval_input({"conversation": conversation}) is True

    def test_missing_query_raises_when_required(self):
        """None query raises MISSING_FIELD when requires_query=True."""
        v = ConversationValidator(error_target=ET, requires_query=True)
        with pytest.raises(EvaluationException) as exc:
            v.validate_eval_input({"query": None, "response": VALID_RESPONSE})
        assert _category(exc) == ErrorCategory.MISSING_FIELD.name

    def test_missing_query_allowed_when_not_required(self):
        """None query is allowed when requires_query=False."""
        v = ConversationValidator(error_target=ET, requires_query=False)
        assert v.validate_eval_input({"response": VALID_RESPONSE}) is True

    def test_missing_response_raises(self):
        """None response raises MISSING_FIELD."""
        v = ConversationValidator(error_target=ET, requires_query=False)
        with pytest.raises(EvaluationException) as exc:
            v.validate_eval_input({"response": None})
        assert _category(exc) == ErrorCategory.MISSING_FIELD.name

    def test_response_wrong_type_raises(self):
        """A non-list / non-string response raises INVALID_VALUE."""
        v = ConversationValidator(error_target=ET, requires_query=False)
        with pytest.raises(EvaluationException) as exc:
            v.validate_eval_input({"response": 123})
        assert _category(exc) == ErrorCategory.INVALID_VALUE.name

    def test_developer_role_accepted(self):
        """Developer role content is validated like user/system (accepted)."""
        v = ConversationValidator(error_target=ET, requires_query=False)
        response = [
            {"role": "developer", "content": [{"type": "text", "text": "system-ish"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ]
        assert v.validate_eval_input({"response": response}) is True

    def test_unsupported_tool_rejected_when_flag_on(self):
        """check_for_unsupported_tools=True rejects an unsupported tool call as NOT_APPLICABLE."""
        v = ConversationValidator(error_target=ET, requires_query=False, check_for_unsupported_tools=True)
        response = [{"role": "assistant", "content": [
            {"type": "tool_call", "name": "bing_grounding", "tool_call_id": "c", "arguments": {}}]}]
        with pytest.raises(EvaluationException) as exc:
            v.validate_eval_input({"response": response})
        assert _category(exc) == ErrorCategory.NOT_APPLICABLE.name

    def test_unsupported_tool_allowed_when_flag_off(self):
        """check_for_unsupported_tools=False allows an otherwise-unsupported tool call."""
        v = ConversationValidator(error_target=ET, requires_query=False, check_for_unsupported_tools=False)
        response = [{"role": "assistant", "content": [
            {"type": "tool_call", "name": "bing_grounding", "tool_call_id": "c", "arguments": {}}]}]
        assert v.validate_eval_input({"response": response}) is True


@pytest.mark.unittest
class TestToolDefinitionsValidator:
    """Unit tests for ToolDefinitionsValidator."""

    def test_valid_with_tool_definitions_passes(self):
        """Valid query/response + valid tool_definitions passes."""
        v = ToolDefinitionsValidator(error_target=ET)
        assert v.validate_eval_input({
            "query": VALID_QUERY, "response": VALID_RESPONSE, "tool_definitions": VALID_TOOL_DEFINITIONS,
        }) is True

    def test_optional_tool_definitions_absent_passes(self):
        """Missing tool_definitions is OK when optional_tool_definitions=True (default)."""
        v = ToolDefinitionsValidator(error_target=ET, optional_tool_definitions=True)
        assert v.validate_eval_input({"query": VALID_QUERY, "response": VALID_RESPONSE}) is True

    def test_required_tool_definitions_absent_raises(self):
        """Missing tool_definitions raises MISSING_FIELD when optional_tool_definitions=False."""
        v = ToolDefinitionsValidator(error_target=ET, optional_tool_definitions=False)
        with pytest.raises(EvaluationException) as exc:
            v.validate_eval_input({"query": VALID_QUERY, "response": VALID_RESPONSE})
        assert _category(exc) == ErrorCategory.MISSING_FIELD.name

    def test_invalid_tool_definition_missing_name_raises(self):
        """A tool definition missing 'name' raises INVALID_VALUE."""
        v = ToolDefinitionsValidator(error_target=ET)
        bad_defs = [{"parameters": {"type": "object", "properties": {}}}]
        with pytest.raises(EvaluationException) as exc:
            v.validate_eval_input({"query": VALID_QUERY, "response": VALID_RESPONSE, "tool_definitions": bad_defs})
        assert _category(exc) == ErrorCategory.INVALID_VALUE.name

    def test_tool_definitions_wrong_type_raises(self):
        """Non-list / non-string tool_definitions raises INVALID_VALUE."""
        v = ToolDefinitionsValidator(error_target=ET)
        with pytest.raises(EvaluationException) as exc:
            v.validate_eval_input({"query": VALID_QUERY, "response": VALID_RESPONSE, "tool_definitions": 123})
        assert _category(exc) == ErrorCategory.INVALID_VALUE.name


@pytest.mark.unittest
class TestToolCallsValidator:
    """Unit tests for ToolCallsValidator."""

    def test_valid_tool_calls_passes(self):
        """Valid query + tool_definitions + tool_calls passes."""
        v = ToolCallsValidator(error_target=ET)
        assert v.validate_eval_input({
            "query": VALID_QUERY, "tool_definitions": VALID_TOOL_DEFINITIONS, "tool_calls": VALID_TOOL_CALLS,
        }) is True

    def test_valid_response_with_tool_calls_in_response_passes(self):
        """Valid response carrying tool calls passes even without separate tool_calls."""
        v = ToolCallsValidator(error_target=ET)
        response = [{"role": "assistant", "content": VALID_TOOL_CALLS}]
        assert v.validate_eval_input({
            "query": VALID_QUERY, "tool_definitions": VALID_TOOL_DEFINITIONS, "response": response,
        }) is True

    def test_missing_response_and_tool_calls_raises(self):
        """Absent response AND tool_calls raises."""
        v = ToolCallsValidator(error_target=ET)
        with pytest.raises(EvaluationException):
            v.validate_eval_input({"query": VALID_QUERY, "tool_definitions": VALID_TOOL_DEFINITIONS})

    def test_missing_tool_definitions_raises(self):
        """ToolCallsValidator requires tool_definitions (optional_tool_definitions=False)."""
        v = ToolCallsValidator(error_target=ET)
        with pytest.raises(EvaluationException):
            v.validate_eval_input({"query": VALID_QUERY, "tool_calls": VALID_TOOL_CALLS})


@pytest.mark.unittest
class TestMessagesOrQueryResponseInputValidator:
    """Unit tests for MessagesOrQueryResponseInputValidator (messages path)."""

    VALID_MESSAGES = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]

    def _validator(self):
        return MessagesOrQueryResponseInputValidator(error_target=ET)

    def test_valid_messages_passes(self):
        """A valid messages list passes."""
        assert self._validator().validate_eval_input({"messages": self.VALID_MESSAGES}) is True

    def test_empty_messages_raises(self):
        """Empty messages list raises INVALID_VALUE."""
        with pytest.raises(EvaluationException) as exc:
            self._validator().validate_eval_input({"messages": []})
        assert _category(exc) == ErrorCategory.INVALID_VALUE.name

    def test_messages_wrong_type_raises(self):
        """Non-list messages raises INVALID_VALUE."""
        with pytest.raises(EvaluationException) as exc:
            self._validator().validate_eval_input({"messages": "not a list"})
        assert _category(exc) == ErrorCategory.INVALID_VALUE.name

    def test_non_dict_message_item_raises(self):
        """A non-dict item in messages raises INVALID_VALUE."""
        messages = [self.VALID_MESSAGES[0], "not a dict", self.VALID_MESSAGES[1]]
        with pytest.raises(EvaluationException) as exc:
            self._validator().validate_eval_input({"messages": messages})
        assert _category(exc) == ErrorCategory.INVALID_VALUE.name

    def test_missing_role_key_raises(self):
        """A message without a role key raises INVALID_VALUE."""
        messages = [self.VALID_MESSAGES[0], {"content": [{"type": "text", "text": "x"}]}, self.VALID_MESSAGES[1]]
        with pytest.raises(EvaluationException) as exc:
            self._validator().validate_eval_input({"messages": messages})
        assert _category(exc) == ErrorCategory.INVALID_VALUE.name

    def test_invalid_role_raises(self):
        """An unknown role raises INVALID_VALUE."""
        messages = [self.VALID_MESSAGES[0], {"role": "narrator", "content": [{"type": "text", "text": "x"}]},
                    self.VALID_MESSAGES[1]]
        with pytest.raises(EvaluationException) as exc:
            self._validator().validate_eval_input({"messages": messages})
        assert _category(exc) == ErrorCategory.INVALID_VALUE.name

    def test_no_user_message_raises(self):
        """Messages without any user role raises INVALID_VALUE."""
        with pytest.raises(EvaluationException) as exc:
            self._validator().validate_eval_input({"messages": [self.VALID_MESSAGES[1]]})
        assert _category(exc) == ErrorCategory.INVALID_VALUE.name

    def test_no_assistant_message_raises(self):
        """Messages without any assistant role raises INVALID_VALUE."""
        with pytest.raises(EvaluationException) as exc:
            self._validator().validate_eval_input({"messages": [self.VALID_MESSAGES[0]]})
        assert _category(exc) == ErrorCategory.INVALID_VALUE.name

    def test_falls_back_to_query_response_when_no_messages(self):
        """When messages is absent, the query/response path is used."""
        v = MessagesOrQueryResponseInputValidator(error_target=ET, requires_query=False)
        assert v.validate_eval_input({"response": VALID_RESPONSE}) is True


@pytest.mark.unittest
class TestTaskNavigationEfficiencyValidator:
    """Unit tests for TaskNavigationEfficiencyValidator."""

    ACTIONS = [{"role": "assistant", "content": [
        {"type": "tool_call", "tool_call_id": "c1", "name": "tool_a", "arguments": {}}]}]
    EXPECTED = ["tool_a"]

    def _validator(self):
        return TaskNavigationEfficiencyValidator(error_target=ET)

    def test_valid_actions_expected_actions_passes(self):
        """Canonical actions/expected_actions pass."""
        assert self._validator().validate_eval_input(
            {"actions": self.ACTIONS, "expected_actions": self.EXPECTED}
        ) is True

    def test_missing_actions_raises(self):
        """Absent actions raises MISSING_FIELD."""
        with pytest.raises(EvaluationException) as exc:
            self._validator().validate_eval_input({"actions": None, "expected_actions": self.EXPECTED})
        assert _category(exc) == ErrorCategory.MISSING_FIELD.name

    def test_empty_expected_actions_raises(self):
        """Empty expected_actions raises MISSING_FIELD."""
        with pytest.raises(EvaluationException) as exc:
            self._validator().validate_eval_input({"actions": self.ACTIONS, "expected_actions": []})
        assert _category(exc) == ErrorCategory.MISSING_FIELD.name

    def test_invalid_expected_actions_type_raises(self):
        """A dict expected_actions raises INVALID_VALUE."""
        with pytest.raises(EvaluationException) as exc:
            self._validator().validate_eval_input({"actions": self.ACTIONS, "expected_actions": {"not": "a list"}})
        assert _category(exc) == ErrorCategory.INVALID_VALUE.name


@pytest.mark.unittest
class TestEnums:
    """Sanity checks for the validator enums."""

    def test_message_role_values(self):
        """MessageRole exposes the expected role string values."""
        values = {r.value for r in MessageRole}
        assert {"user", "assistant", "system", "tool", "developer"}.issubset(values)

    def test_content_type_values(self):
        """ContentType exposes core content type values."""
        values = {c.value for c in ContentType}
        assert {"text", "tool_call", "tool_result"}.issubset(values)
