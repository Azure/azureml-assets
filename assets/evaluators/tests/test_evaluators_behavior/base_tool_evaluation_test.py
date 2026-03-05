# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for testing evaluators with specific tool types (function, built-in, etc.).

Contains ALL tool-type test methods so that evaluator-specific derivative classes
only need to set evaluator_type (and potentially expected_result_fields).
Uses the base class _run_evaluation_and_return_mocked_flow with use_mocking=True
and asserts correct behavior using assert_expected_behavior and assert_called_once_with.
"""

from typing import Any, Dict, Optional

from . import common_tool_test_data as data
from ..common.base_prompty_evaluator_runner import BasePromptyEvaluatorRunner


class BaseToolEvaluationTest(BasePromptyEvaluatorRunner):
    """
    Base class for tool-type-specific evaluator tests.

    Contains all tool-type test methods. Subclasses should only set:
    - evaluator_type: The evaluator class to test

    Subclasses may override:
    - expected_result_fields: list of expected fields in evaluation results
    - check_for_unsupported_tools: whether to check for unsupported tool types in the conversation (default False since many tool types not supported by Tool Call Accuracy Evaluator)
    """

    use_mocking = True

    check_for_unsupported_tools: bool = False

    is_tool_definition_required: bool = False

    #region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {}

    test_code_interpreter_expected_flow_inputs = {}

    test_bing_grounding_expected_flow_inputs = {}

    test_bing_custom_search_expected_flow_inputs = {}

    test_file_search_expected_flow_inputs = {}

    test_azure_ai_search_expected_flow_inputs = {}

    test_sharepoint_grounding_expected_flow_inputs = {}

    test_fabric_data_agent_expected_flow_inputs = {}

    test_openapi_expected_flow_inputs = {}

    test_web_search_expected_flow_inputs = {}

    test_browser_automation_expected_flow_inputs = {}

    test_computer_use_expected_flow_inputs = {}

    test_image_generation_expected_flow_inputs = {}

    test_memory_search_expected_flow_inputs = {}

    test_kb_mcp_expected_flow_inputs = {}

    test_mcp_expected_flow_inputs = {}
    #endregion

    def _run_tool_type_test(
        self,
        *,
        test_label: str,
        evaluation_inputs: Dict[str, Any],
        assert_type: BasePromptyEvaluatorRunner.AssertType,
        expected_flow_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a tool-type test, assert expected behavior, and verify flow mock.

        Args:
            test_label: Descriptive label for the test.
            evaluation_inputs: Dictionary containing query, response, tool_definitions, and optionally tool_calls, context.
            assert_type: Expected behavior (PASS, INVALID_VALUE, MISSING_FIELD, NOT_APPLICABLE).
            expected_flow_inputs: Optional dictionary of expected inputs to the flow (query, tool_calls, tool_definitions).

        Returns:
            Dictionary containing the extracted result data.
        """
        results, flow_mock = self._run_evaluation_and_return_mocked_flow(
            **evaluation_inputs,
        )
        result_data = self._extract_and_print_result(results, test_label)
        self.assert_expected_behavior(assert_type, result_data)

        # Assert flow mock behavior
        # Flow is called if and only if assert_type is PASS
        expected_flow_called = assert_type == self.AssertType.PASS
        assert flow_mock is not None, "Flow mock should be set when use_mocking=True"
        if expected_flow_called:
            flow_mock.assert_called_once_with(
                timeout=600,
                **expected_flow_inputs,
            )
        else:
            flow_mock.assert_not_called()

        return result_data

    # ==================== TOOL TYPE TESTS ====================

    # --- Function Tool (local_calls) ---

    def test_function_tool_local_calls(self):
        """Function tool (get_horoscope) with function_call content type."""
        self._run_tool_type_test(
            test_label="Function Tool - local_calls",
            evaluation_inputs={
                "query": data.LOCAL_CALLS_QUERY,
                "response": data.LOCAL_CALLS_RESPONSE,
                "tool_definitions": data.LOCAL_CALLS_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.PASS,
            expected_flow_inputs=self.test_function_tool_local_calls_expected_flow_inputs,
        )

    # --- Code Interpreter ---

    def test_code_interpreter(self):
        """Code interpreter tool with code_interpreter type - not supported."""
        self._run_tool_type_test(
            test_label="Code Interpreter",
            evaluation_inputs={
                "query": data.CODE_INTERPRETER_QUERY,
                "response": data.CODE_INTERPRETER_RESPONSE,
                "tool_definitions": data.CODE_INTERPRETER_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.NOT_APPLICABLE if self.check_for_unsupported_tools else self.AssertType.PASS,
            expected_flow_inputs=self.test_code_interpreter_expected_flow_inputs,
        )

    # --- Bing Grounding ---

    def test_bing_grounding(self):
        """Bing grounding tool with remote_tool type - not supported."""
        self._run_tool_type_test(
            test_label="Bing Grounding",
            evaluation_inputs={
                "query": data.BING_GROUNDING_QUERY,
                "response": data.BING_GROUNDING_RESPONSE,
                "tool_definitions": data.BING_GROUNDING_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.NOT_APPLICABLE if self.check_for_unsupported_tools else self.AssertType.PASS,
            expected_flow_inputs=self.test_bing_grounding_expected_flow_inputs,
        )

    # --- Bing Custom Search ---

    def test_bing_custom_search(self):
        """Bing custom search tool with remote_tool type - not supported."""
        self._run_tool_type_test(
            test_label="Bing Custom Search",
            evaluation_inputs={
                "query": data.BING_CUSTOM_SEARCH_QUERY,
                "response": data.BING_CUSTOM_SEARCH_RESPONSE,
                "tool_definitions": data.BING_CUSTOM_SEARCH_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.NOT_APPLICABLE if self.check_for_unsupported_tools else self.AssertType.PASS,
            expected_flow_inputs=self.test_bing_custom_search_expected_flow_inputs,
        )

    # --- File Search ---

    def test_file_search(self):
        """File search tool with file_search type."""
        self._run_tool_type_test(
            test_label="File Search",
            evaluation_inputs={
                "query": data.FILE_SEARCH_QUERY,
                "response": data.FILE_SEARCH_RESPONSE,
                "tool_definitions": data.FILE_SEARCH_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.PASS,
            expected_flow_inputs=self.test_file_search_expected_flow_inputs,
        )

    # --- Azure AI Search ---

    def test_azure_ai_search(self):
        """Azure AI Search tool with remote_tool type - not supported."""
        self._run_tool_type_test(
            test_label="Azure AI Search",
            evaluation_inputs={
                "query": data.AZURE_AI_SEARCH_QUERY,
                "response": data.AZURE_AI_SEARCH_RESPONSE,
                "tool_definitions": data.AZURE_AI_SEARCH_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.NOT_APPLICABLE if self.check_for_unsupported_tools else self.AssertType.PASS,
            expected_flow_inputs=self.test_azure_ai_search_expected_flow_inputs,
        )

    # --- SharePoint Grounding ---

    def test_sharepoint_grounding(self):
        """SharePoint grounding tool with remote_tool type - not supported."""
        self._run_tool_type_test(
            test_label="SharePoint Grounding",
            evaluation_inputs={
                "query": data.SHAREPOINT_QUERY,
                "response": data.SHAREPOINT_RESPONSE,
                "tool_definitions": data.SHAREPOINT_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.NOT_APPLICABLE if self.check_for_unsupported_tools else self.AssertType.PASS,
            expected_flow_inputs=self.test_sharepoint_grounding_expected_flow_inputs,
        )

    # --- Fabric Data Agent ---

    def test_fabric_data_agent(self):
        """Fabric data agent tool with remote_tool type - not supported."""
        self._run_tool_type_test(
            test_label="Fabric Data Agent",
            evaluation_inputs={
                "query": data.FABRIC_QUERY,
                "response": data.FABRIC_RESPONSE,
                "tool_definitions": data.FABRIC_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.NOT_APPLICABLE if self.check_for_unsupported_tools else self.AssertType.PASS,
            expected_flow_inputs=self.test_fabric_data_agent_expected_flow_inputs,
        )

    # --- OpenAPI ---

    def test_openapi(self):
        """OpenAPI tool with openapi_call content type - invalid without 'functions' field."""
        self._run_tool_type_test(
            test_label="OpenAPI",
            evaluation_inputs={
                "query": data.OPENAPI_QUERY,
                "response": data.OPENAPI_RESPONSE,
                "tool_definitions": data.OPENAPI_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.NOT_APPLICABLE if self.check_for_unsupported_tools else self.AssertType.PASS,
            expected_flow_inputs=self.test_openapi_expected_flow_inputs,
        )

    # --- Web Search ---

    def test_web_search(self):
        """Web search tool with web_search_preview type - not supported."""
        self._run_tool_type_test(
            test_label="Web Search",
            evaluation_inputs={
                "query": data.WEB_SEARCH_QUERY,
                "response": data.WEB_SEARCH_RESPONSE,
                "tool_definitions": data.WEB_SEARCH_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.NOT_APPLICABLE if self.check_for_unsupported_tools else self.AssertType.PASS,
            expected_flow_inputs=self.test_web_search_expected_flow_inputs,
        )

    # --- Browser Automation ---

    def test_browser_automation(self):
        """Browser automation tool with remote_tool type - not supported."""
        self._run_tool_type_test(
            test_label="Browser Automation",
            evaluation_inputs={
                "query": data.BROWSER_AUTOMATION_QUERY,
                "response": data.BROWSER_AUTOMATION_RESPONSE,
                "tool_definitions": data.BROWSER_AUTOMATION_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.NOT_APPLICABLE if self.check_for_unsupported_tools else self.AssertType.PASS,
            expected_flow_inputs=self.test_browser_automation_expected_flow_inputs,
        )

    # --- Computer Use ---

    def test_computer_use(self):
        """Computer use tool with empty tool_definitions - missing field error."""
        self._run_tool_type_test(
            test_label="Computer Use",
            evaluation_inputs={
                "query": data.COMPUTER_USE_QUERY,
                "response": data.COMPUTER_USE_RESPONSE,
                "tool_definitions": data.COMPUTER_USE_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.MISSING_FIELD if self.is_tool_definition_required else self.AssertType.INVALID_VALUE,
            expected_flow_inputs=self.test_computer_use_expected_flow_inputs,
        )

    # --- Image Generation ---

    def test_image_generation(self):
        """Image generation tool with image_generation type."""
        self._run_tool_type_test(
            test_label="Image Generation",
            evaluation_inputs={
                "query": data.IMAGE_GEN_QUERY,
                "response": data.IMAGE_GEN_RESPONSE,
                "tool_definitions": data.IMAGE_GEN_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.PASS,
            expected_flow_inputs=self.test_image_generation_expected_flow_inputs,
        )

    # --- Memory Search ---

    def test_memory_search(self):
        """Memory search tool with memory_search type."""
        self._run_tool_type_test(
            test_label="Memory Search",
            evaluation_inputs={
                "query": data.MEMORY_SEARCH_QUERY,
                "response": data.MEMORY_SEARCH_RESPONSE,
                "tool_definitions": data.MEMORY_SEARCH_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.PASS,
            expected_flow_inputs=self.test_memory_search_expected_flow_inputs,
        )

    # --- Knowledge Base MCP ---

    def test_kb_mcp(self):
        """Knowledge base with MCP approval request/response."""
        self._run_tool_type_test(
            test_label="KB MCP",
            evaluation_inputs={
                "query": data.KB_MCP_QUERY,
                "response": data.KB_MCP_RESPONSE,
                "tool_definitions": data.KB_MCP_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.PASS,
            expected_flow_inputs=self.test_kb_mcp_expected_flow_inputs,
        )

    # --- MCP (Multi-tool) ---

    def test_mcp(self):
        """MCP with multiple tool types (code_interpreter, web_search, function)."""
        self._run_tool_type_test(
            test_label="MCP Multi-tool",
            evaluation_inputs={
                "query": data.MCP_QUERY,
                "response": data.MCP_RESPONSE,
                "tool_definitions": data.MCP_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.PASS,
            expected_flow_inputs=self.test_mcp_expected_flow_inputs,
        )
