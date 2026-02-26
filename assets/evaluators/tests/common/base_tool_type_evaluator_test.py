# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for testing evaluators with specific tool types (function, built-in, etc.).

Contains ALL tool-type test methods so that evaluator-specific derivative classes
only need to set evaluator_type (and potentially expected_result_fields).
Uses the base class _run_evaluation_and_return_mocked_flow with use_mocking=True
and asserts correct behavior using assert_expected_behavior and assert_called_once_with.
"""

from typing import Any, Dict, List

from ..test_evaluators_tools import tool_type_test_data as data
from .base_prompty_evaluator_runner import BasePromptyEvaluatorRunner


class BaseToolTypeEvaluatorTest(BasePromptyEvaluatorRunner):
    """
    Base class for tool-type-specific evaluator tests.

    Contains all tool-type test methods. Subclasses should only set:
    - evaluator_type: The evaluator class to test

    Subclasses may override:
    - expected_result_fields: list of expected fields in evaluation results
    """

    use_mocking = True

    _additional_expected_field_suffixes = ["details"]

    @property
    def expected_result_fields(self) -> List[str]:
        """Get the expected result fields for tool type evaluator tests."""
        return [
            f"{self._result_prefix}",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_threshold",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_prompt_tokens",
            f"{self._result_prefix}_completion_tokens",
            f"{self._result_prefix}_total_tokens",
            f"{self._result_prefix}_finish_reason",
            f"{self._result_prefix}_model",
            f"{self._result_prefix}_sample_input",
            f"{self._result_prefix}_sample_output",
        ] + [f"{self._result_prefix}_{suffix}" for suffix in self._additional_expected_field_suffixes]

    def _run_tool_type_test(
        self,
        *,
        test_label: str,
        query: Any,
        response: Any,
        tool_definitions: Any,
        assert_type: BasePromptyEvaluatorRunner.AssertType,
        tool_calls: Any = None,
        context: str = None,
        expected_flow_called: bool,
        expected_flow_tool_calls: Any = None,
        expected_flow_tool_definitions: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run a tool-type test, assert expected behavior, and verify flow mock.

        Args:
            test_label: Descriptive label for the test.
            query: Query input (query_history_list from converter output).
            response: Response input (response from converter output).
            tool_definitions: Tool definitions from converter output.
            assert_type: Expected behavior (PASS, INVALID_VALUE, MISSING_FIELD, NOT_APPLICABLE).
            tool_calls: Tool calls data (optional).
            context: Additional context (optional).
            expected_flow_called: Whether the flow mock should have been called.
            expected_flow_tool_calls: Expected tool_calls arg passed to flow (required when expected_flow_called=True).
            expected_flow_tool_definitions: Expected tool_definitions arg passed to flow (defaults to tool_definitions).
            **kwargs: Additional keyword arguments passed to _run_evaluation.

        Returns:
            Dictionary containing the extracted result data.
        """
        results, flow_mock = self._run_evaluation_and_return_mocked_flow(
            query=query,
            response=response,
            tool_definitions=tool_definitions,
            tool_calls=tool_calls,
            context=context,
            **kwargs,
        )
        result_data = self._extract_and_print_result(results, test_label)
        self.assert_expected_behavior(assert_type, result_data)

        # Assert flow mock behavior
        assert flow_mock is not None, "Flow mock should be set when use_mocking=True"
        if expected_flow_called:
            flow_mock.assert_called_once_with(
                timeout=600,
                query=query,
                tool_calls=expected_flow_tool_calls,
                tool_definitions=expected_flow_tool_definitions if expected_flow_tool_definitions is not None else tool_definitions,
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
            query=data.LOCAL_CALLS_QUERY,
            response=data.LOCAL_CALLS_RESPONSE,
            tool_definitions=data.LOCAL_CALLS_TOOL_DEFINITIONS,
            assert_type=self.AssertType.PASS,
            expected_flow_called=True,
            expected_flow_tool_calls=data.LOCAL_CALLS_EXPECTED_FLOW_TOOL_CALLS,
        )

    # --- Code Interpreter ---

    def test_code_interpreter(self):
        """Code interpreter tool with code_interpreter type - not supported."""
        self._run_tool_type_test(
            test_label="Code Interpreter",
            query=data.CODE_INTERPRETER_QUERY,
            response=data.CODE_INTERPRETER_RESPONSE,
            tool_definitions=data.CODE_INTERPRETER_TOOL_DEFINITIONS,
            assert_type=self.AssertType.NOT_APPLICABLE,
            expected_flow_called=False,
        )

    # --- Bing Grounding ---

    def test_bing_grounding(self):
        """Bing grounding tool with remote_tool type - not supported."""
        self._run_tool_type_test(
            test_label="Bing Grounding",
            query=data.BING_GROUNDING_QUERY,
            response=data.BING_GROUNDING_RESPONSE,
            tool_definitions=data.BING_GROUNDING_TOOL_DEFINITIONS,
            assert_type=self.AssertType.NOT_APPLICABLE,
            expected_flow_called=False,
        )

    # --- Bing Custom Search ---

    def test_bing_custom_search(self):
        """Bing custom search tool with remote_tool type - not supported."""
        self._run_tool_type_test(
            test_label="Bing Custom Search",
            query=data.BING_CUSTOM_SEARCH_QUERY,
            response=data.BING_CUSTOM_SEARCH_RESPONSE,
            tool_definitions=data.BING_CUSTOM_SEARCH_TOOL_DEFINITIONS,
            assert_type=self.AssertType.NOT_APPLICABLE,
            expected_flow_called=False,
        )

    # --- File Search ---

    def test_file_search(self):
        """File search tool with file_search type."""
        self._run_tool_type_test(
            test_label="File Search",
            query=data.FILE_SEARCH_QUERY,
            response=data.FILE_SEARCH_RESPONSE,
            tool_definitions=data.FILE_SEARCH_TOOL_DEFINITIONS,
            assert_type=self.AssertType.PASS,
            expected_flow_called=True,
            expected_flow_tool_calls=data.FILE_SEARCH_EXPECTED_FLOW_TOOL_CALLS,
        )

    # --- Azure AI Search ---

    def test_azure_ai_search(self):
        """Azure AI Search tool with remote_tool type - not supported."""
        self._run_tool_type_test(
            test_label="Azure AI Search",
            query=data.AZURE_AI_SEARCH_QUERY,
            response=data.AZURE_AI_SEARCH_RESPONSE,
            tool_definitions=data.AZURE_AI_SEARCH_TOOL_DEFINITIONS,
            assert_type=self.AssertType.NOT_APPLICABLE,
            expected_flow_called=False,
        )

    # --- SharePoint Grounding ---

    def test_sharepoint_grounding(self):
        """SharePoint grounding tool with remote_tool type - not supported."""
        self._run_tool_type_test(
            test_label="SharePoint Grounding",
            query=data.SHAREPOINT_QUERY,
            response=data.SHAREPOINT_RESPONSE,
            tool_definitions=data.SHAREPOINT_TOOL_DEFINITIONS,
            assert_type=self.AssertType.NOT_APPLICABLE,
            expected_flow_called=False,
        )

    # --- Fabric Data Agent ---

    def test_fabric_data_agent(self):
        """Fabric data agent tool with remote_tool type - not supported."""
        self._run_tool_type_test(
            test_label="Fabric Data Agent",
            query=data.FABRIC_QUERY,
            response=data.FABRIC_RESPONSE,
            tool_definitions=data.FABRIC_TOOL_DEFINITIONS,
            assert_type=self.AssertType.NOT_APPLICABLE,
            expected_flow_called=False,
        )

    # --- OpenAPI ---

    def test_openapi(self):
        """OpenAPI tool with openapi_call content type - invalid without 'functions' field."""
        self._run_tool_type_test(
            test_label="OpenAPI",
            query=data.OPENAPI_QUERY,
            response=data.OPENAPI_RESPONSE,
            tool_definitions=data.OPENAPI_TOOL_DEFINITIONS,
            assert_type=self.AssertType.INVALID_VALUE,
            expected_flow_called=False,
        )

    # --- Web Search ---

    def test_web_search(self):
        """Web search tool with web_search_preview type - not supported."""
        self._run_tool_type_test(
            test_label="Web Search",
            query=data.WEB_SEARCH_QUERY,
            response=data.WEB_SEARCH_RESPONSE,
            tool_definitions=data.WEB_SEARCH_TOOL_DEFINITIONS,
            assert_type=self.AssertType.NOT_APPLICABLE,
            expected_flow_called=False,
        )

    # --- Browser Automation ---

    def test_browser_automation(self):
        """Browser automation tool with remote_tool type - not supported."""
        self._run_tool_type_test(
            test_label="Browser Automation",
            query=data.BROWSER_AUTOMATION_QUERY,
            response=data.BROWSER_AUTOMATION_RESPONSE,
            tool_definitions=data.BROWSER_AUTOMATION_TOOL_DEFINITIONS,
            assert_type=self.AssertType.NOT_APPLICABLE,
            expected_flow_called=False,
        )

    # --- Computer Use ---

    def test_computer_use(self):
        """Computer use tool with empty tool_definitions - missing field error."""
        self._run_tool_type_test(
            test_label="Computer Use",
            query=data.COMPUTER_USE_QUERY,
            response=data.COMPUTER_USE_RESPONSE,
            tool_definitions=data.COMPUTER_USE_TOOL_DEFINITIONS,
            assert_type=self.AssertType.MISSING_FIELD,
            expected_flow_called=False,
        )

    # --- Image Generation ---

    def test_image_generation(self):
        """Image generation tool with image_generation type."""
        self._run_tool_type_test(
            test_label="Image Generation",
            query=data.IMAGE_GEN_QUERY,
            response=data.IMAGE_GEN_RESPONSE,
            tool_definitions=data.IMAGE_GEN_TOOL_DEFINITIONS,
            assert_type=self.AssertType.PASS,
            expected_flow_called=True,
            expected_flow_tool_calls=data.IMAGE_GEN_EXPECTED_FLOW_TOOL_CALLS,
        )

    # --- Memory Search ---

    def test_memory_search(self):
        """Memory search tool with memory_search type."""
        self._run_tool_type_test(
            test_label="Memory Search",
            query=data.MEMORY_SEARCH_QUERY,
            response=data.MEMORY_SEARCH_RESPONSE,
            tool_definitions=data.MEMORY_SEARCH_TOOL_DEFINITIONS,
            assert_type=self.AssertType.PASS,
            expected_flow_called=True,
            expected_flow_tool_calls=data.MEMORY_SEARCH_EXPECTED_FLOW_TOOL_CALLS,
        )

    # --- Knowledge Base MCP ---

    def test_kb_mcp(self):
        """Knowledge base with MCP approval request/response."""
        self._run_tool_type_test(
            test_label="KB MCP",
            query=data.KB_MCP_QUERY,
            response=data.KB_MCP_RESPONSE,
            tool_definitions=data.KB_MCP_TOOL_DEFINITIONS,
            assert_type=self.AssertType.PASS,
            expected_flow_called=True,
            expected_flow_tool_calls=data.KB_MCP_EXPECTED_FLOW_TOOL_CALLS,
        )

    # --- MCP (Multi-tool) ---

    def test_mcp(self):
        """MCP with multiple tool types (code_interpreter, web_search, function)."""
        self._run_tool_type_test(
            test_label="MCP Multi-tool",
            query=data.MCP_QUERY,
            response=data.MCP_RESPONSE,
            tool_definitions=data.MCP_TOOL_DEFINITIONS,
            assert_type=self.AssertType.PASS,
            expected_flow_called=True,
            expected_flow_tool_calls=data.MCP_EXPECTED_FLOW_TOOL_CALLS,
        )
