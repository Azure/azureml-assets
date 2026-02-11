# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Response Completeness Evaluator."""

import pytest
import math
from ..common.base_prompty_evaluator_runner import BasePromptyEvaluatorRunner
from ...builtin.response_completeness.evaluator._response_completeness import ResponseCompletenessEvaluator


@pytest.mark.unittest
class TestResponseCompletenessEvaluatorBehavior(BasePromptyEvaluatorRunner):
    """
    Behavioral tests for Response Completeness Evaluator.

    Tests different input formats, error handling, and edge cases.
    Response Completeness evaluator requires both 'response' and 'ground_truth' as string inputs.
    """

    evaluator_type = ResponseCompletenessEvaluator

    constructor_arg_names = ["threshold"]

    # ==================== VALID INPUT TESTS ====================

    def test_valid_string_inputs(self) -> None:
        """Test case: Valid string inputs for response and ground_truth.

        Both response and ground_truth are provided as strings.
        """
        results = self._run_evaluation(
            response="Python is a programming language created by Guido van Rossum.",
            ground_truth="Python is a programming language created by Guido van Rossum.",
        )

        result_data = self._extract_and_print_result(results, "valid-string-inputs")
        self.assert_pass(result_data)

    def test_valid_different_string_lengths(self) -> None:
        """Test case: Valid inputs with different string lengths.

        Ground truth is longer than response.
        """
        results = self._run_evaluation(
            response="Water boils at 100 degrees Celsius.",
            ground_truth=(
                "Water boils at 100 degrees Celsius at sea level atmospheric pressure. "
                "At higher altitudes, the boiling point decreases due to lower pressure."
            ),
        )

        result_data = self._extract_and_print_result(results, "valid-different-string-lengths")
        self.assert_pass_or_fail(result_data)

    def test_valid_response_longer_than_ground_truth(self) -> None:
        """Test case: Valid inputs where response is longer.

        Response contains additional information beyond ground truth.
        """
        results = self._run_evaluation(
            ground_truth="The Earth is round.",
            response=(
                "The Earth is round, actually an oblate spheroid. This means it's slightly "
                "flattened at the poles and bulging at the equator due to its rotation."
            ),
        )

        result_data = self._extract_and_print_result(results, "valid-response-longer")
        self.assert_pass(result_data)

    def test_valid_minimal_strings(self) -> None:
        """Test case: Valid minimal string inputs.

        Both inputs are very short but valid.
        """
        results = self._run_evaluation(
            response="Paris",
            ground_truth="Paris",
        )

        result_data = self._extract_and_print_result(results, "valid-minimal-strings")
        self.assert_pass(result_data)

    def test_valid_multiline_strings(self) -> None:
        """Test case: Valid inputs with multiline strings.

        Tests handling of newlines and multiple paragraphs.
        """
        results = self._run_evaluation(
            response=(
                "Line one contains information.\n"
                "Line two has more details.\n\n"
                "A new paragraph starts here."
            ),
            ground_truth=(
                "Line one contains information.\n"
                "Line two has more details.\n\n"
                "A new paragraph starts here."
            ),
        )

        result_data = self._extract_and_print_result(results, "valid-multiline-strings")
        self.assert_pass(result_data)

    def test_valid_special_characters(self) -> None:
        """Test case: Valid inputs with special characters.

        Tests handling of punctuation, symbols, and special characters.
        """
        results = self._run_evaluation(
            response="E = mc²! Einstein's equation (1905) revolutionized physics.",
            ground_truth="E = mc² is Einstein's equation from 1905 that revolutionized physics.",
        )

        result_data = self._extract_and_print_result(results, "valid-special-characters")
        self.assert_pass(result_data)

    def test_valid_unicode_characters(self) -> None:
        """Test case: Valid inputs with unicode characters.

        Tests handling of non-ASCII characters.
        """
        results = self._run_evaluation(
            response="Tokyo (東京) is the capital of Japan. It's a major city.",
            ground_truth="Tokyo (東京) is Japan's capital and a major metropolitan area.",
        )

        result_data = self._extract_and_print_result(results, "valid-unicode-characters")
        self.assert_pass(result_data)

    # ==================== MISSING FIELD TESTS ====================

    def test_missing_response_field(self) -> None:
        """Test case: Missing response field.

        Only ground_truth is provided, response is missing.
        """
        results = self._run_evaluation(
            ground_truth="This is the ground truth.",
        )

        result_data = self._extract_and_print_result(results, "missing-response-field")
        # TODO: Should this be a missing field error
        self.assert_invalid_value_error(result_data)

    def test_missing_ground_truth_field(self) -> None:
        """Test case: Missing ground_truth field.

        Only response is provided, ground_truth is missing.
        """
        results = self._run_evaluation(
            response="This is the response.",
        )

        result_data = self._extract_and_print_result(results, "missing-ground-truth-field")
        # TODO: Should this be a missing field error
        self.assert_invalid_value_error(result_data)

    def test_missing_both_fields(self) -> None:
        """Test case: Both response and ground_truth missing.

        Neither required field is provided.
        """
        results = self._run_evaluation()

        result_data = self._extract_and_print_result(results, "missing-both-fields")
        # TODO: Should this be a missing field error for both fields
        self.assert_invalid_value_error(result_data)

    # ==================== EMPTY STRING TESTS ====================

    def test_empty_response_string(self) -> None:
        """Test case: Empty response string.

        Response is provided but is an empty string.
        """
        results = self._run_evaluation(
            response="",
            ground_truth="This is the ground truth with content.",
        )

        result_data = self._extract_and_print_result(results, "empty-response-string")
        self.assert_pass_or_fail(result_data)

    def test_empty_ground_truth_string(self) -> None:
        """Test case: Empty ground_truth string.

        Ground truth is provided but is an empty string.
        """
        results = self._run_evaluation(
            response="This is a response with content.",
            ground_truth="",
        )

        result_data = self._extract_and_print_result(results, "empty-ground-truth-string")
        self.assert_pass_or_fail(result_data)

    def test_both_empty_strings(self) -> None:
        """Test case: Both response and ground_truth are empty.

        Both fields provided but contain empty strings.
        """
        results = self._run_evaluation(
            response="",
            ground_truth="",
        )

        result_data = self._extract_and_print_result(results, "both-empty-strings")
        self.assert_pass_or_fail(result_data)

    def test_whitespace_only_response(self) -> None:
        """Test case: Response contains only whitespace.

        Response is whitespace characters only.
        """
        results = self._run_evaluation(
            response="   \n\t  ",
            ground_truth="This is actual content.",
        )

        result_data = self._extract_and_print_result(results, "whitespace-only-response")
        self.assert_pass_or_fail(result_data)

    def test_whitespace_only_ground_truth(self) -> None:
        """Test case: Ground truth contains only whitespace.

        Ground truth is whitespace characters only.
        """
        results = self._run_evaluation(
            response="This is actual content.",
            ground_truth="   \n\t  ",
        )

        result_data = self._extract_and_print_result(results, "whitespace-only-ground-truth")
        self.assert_pass_or_fail(result_data)

    # ==================== THRESHOLD TESTS ====================

    def test_custom_threshold_below_default(self) -> None:
        """Test case: Custom threshold below default.

        Sets threshold to 2 (default is 3).
        """
        results = self._run_evaluation(
            response="Partial information only.",
            ground_truth="Complete information with many details that are comprehensive.",
            threshold=2,
        )

        result_data = self._extract_and_print_result(results, "custom-threshold-below-default")
        assert result_data["threshold"] == 2

    def test_custom_threshold_above_default(self) -> None:
        """Test case: Custom threshold above default.

        Sets threshold to 4 (default is 3).
        """
        results = self._run_evaluation(
            response="Very complete response with most details.",
            ground_truth="Complete information with all details.",
            threshold=4,
        )

        result_data = self._extract_and_print_result(results, "custom-threshold-above-default")
        assert result_data["threshold"] == 4

    def test_threshold_at_minimum_score(self) -> None:
        """Test case: Threshold at minimum score boundary.

        Sets threshold to 1 (minimum possible).
        """
        results = self._run_evaluation(
            response="Some response.",
            ground_truth="Some ground truth.",
            threshold=1,
        )

        result_data = self._extract_and_print_result(results, "threshold-at-minimum")
        assert result_data["threshold"] == 1

    def test_threshold_at_maximum_score(self) -> None:
        """Test case: Threshold at maximum score boundary.

        Sets threshold to 5 (maximum possible).
        """
        results = self._run_evaluation(
            response="Perfect complete response.",
            ground_truth="Perfect complete ground truth.",
            threshold=5,
        )

        result_data = self._extract_and_print_result(results, "threshold-at-maximum")
        assert result_data["threshold"] == 5

    # ==================== RESULT FORMAT TESTS ====================

    def test_result_contains_expected_fields(self) -> None:
        """Test case: Result contains all expected fields.

        Validates that the result dictionary has all required fields.
        """
        results = self._run_evaluation(
            response="The Moon orbits the Earth.",
            ground_truth="The Moon orbits the Earth.",
        )

        # Check all expected fields are present
        assert "response_completeness" in results
        assert "response_completeness_reason" in results
        assert "response_completeness_result" in results
        assert "response_completeness_threshold" in results

    def test_score_is_integer_type(self) -> None:
        """Test case: Score is returned as integer.

        Validates that score field contains an integer value (1-5).
        """
        results = self._run_evaluation(
            response="Information here.",
            ground_truth="Information here.",
        )

        score = results.get("response_completeness")
        # Score should be an integer between 1 and 5, or NaN
        assert isinstance(score, (int, float))
        if not math.isnan(score):
            assert isinstance(score, int)
            assert 1 <= score <= 5

    def test_reason_field_is_string(self) -> None:
        """Test case: Reason field contains string.

        Validates that reason field is a string.
        """
        results = self._run_evaluation(
            response="Test response.",
            ground_truth="Test ground truth.",
        )

        reason = results.get("response_completeness_reason")
        assert reason is None or isinstance(reason, str)

    def test_result_field_is_pass_or_fail(self) -> None:
        """Test case: Result field is 'pass' or 'fail'.

        Validates that result field contains expected values.
        """
        results = self._run_evaluation(
            response="Complete response.",
            ground_truth="Complete ground truth.",
        )

        result = results.get("response_completeness_result")
        assert result in ["pass", "fail"]

    # ==================== EDGE CASE TESTS ====================

    def test_response_is_substring_of_ground_truth(self) -> None:
        """Test case: Response is exact substring of ground truth.

        Tests partial completeness when response is contained in ground truth.
        """
        ground_truth_text = "The quick brown fox jumps over the lazy dog."
        response_text = "The quick brown fox"

        results = self._run_evaluation(
            response=response_text,
            ground_truth=ground_truth_text,
        )

        result_data = self._extract_and_print_result(results, "response-is-substring")
        self.assert_pass_or_fail(result_data)

    def test_ground_truth_is_substring_of_response(self) -> None:
        """Test case: Ground truth is substring of response.

        Tests when response contains all of ground truth plus extra.
        """
        ground_truth_text = "Water is H2O."
        response_text = "Water is H2O. It consists of two hydrogen atoms and one oxygen atom."

        results = self._run_evaluation(
            response=response_text,
            ground_truth=ground_truth_text,
        )

        result_data = self._extract_and_print_result(results, "ground-truth-is-substring")
        self.assert_pass(result_data)

    def test_identical_strings_different_whitespace(self) -> None:
        """Test case: Identical content with different whitespace.

        Tests whether extra spaces/newlines affect completeness scoring.
        """
        results = self._run_evaluation(
            response="The cat sat on the mat.",
            ground_truth="The  cat  sat  on  the  mat.",
        )

        result_data = self._extract_and_print_result(results, "identical-different-whitespace")
        self.assert_pass(result_data)

    def test_response_with_html_tags(self) -> None:
        """Test case: Response contains HTML tags.

        Tests handling of markup in response text.
        """
        results = self._run_evaluation(
            response="<p>The <strong>Earth</strong> is round.</p>",
            ground_truth="The Earth is round.",
        )

        result_data = self._extract_and_print_result(results, "response-with-html-tags")
        self.assert_pass_or_fail(result_data)

    def test_response_with_code_snippets(self) -> None:
        """Test case: Response contains code snippets.

        Tests handling of code in response text.
        """
        results = self._run_evaluation(
            response="To print in Python, use: print('Hello')",
            ground_truth="To print in Python, use the print function with your text.",
        )

        result_data = self._extract_and_print_result(results, "response-with-code-snippets")
        self.assert_pass_or_fail(result_data)

    def test_ground_truth_with_citations(self) -> None:
        """Test case: Ground truth contains citations.

        Tests handling of reference citations in text.
        """
        results = self._run_evaluation(
            response="Climate change is affecting global temperatures.",
            ground_truth="Climate change is affecting global temperatures [1].",
        )

        result_data = self._extract_and_print_result(results, "ground-truth-with-citations")
        self.assert_pass(result_data)

    def test_numbers_in_different_formats(self) -> None:
        """Test case: Numbers represented in different formats.

        Tests whether numeric equivalence is recognized.
        """
        results = self._run_evaluation(
            response="The temperature is 25 degrees Celsius.",
            ground_truth="The temperature is twenty-five degrees Celsius.",
        )

        result_data = self._extract_and_print_result(results, "numbers-different-formats")
        self.assert_pass_or_fail(result_data)

    def test_abbreviations_vs_full_words(self) -> None:
        """Test case: Abbreviations versus full words.

        Tests handling of abbreviated versus spelled-out terms.
        """
        results = self._run_evaluation(
            response="The USA is in North America.",
            ground_truth="The United States of America is in North America.",
        )

        result_data = self._extract_and_print_result(results, "abbreviations-vs-full-words")
        self.assert_pass(result_data)

    # ==================== TYPE HANDLING TESTS ====================
    # TODO: Decide whether evaluator should fail on wrong types or handle gracefully.
    # Currently these tests expect the evaluator to handle type conversion gracefully (pass/fail result).
    # If strict type validation is desired, these should be changed to expect errors.

    def test_response_as_non_string_type(self) -> None:
        """Test case: Response provided as non-string type.

        Tests handling when response is not a string. Currently expects graceful handling
        (e.g., automatic string conversion).

        TODO: Decide if this should raise an error instead.
        """
        results = self._run_evaluation(
            response=12345,  # Integer instead of string
            ground_truth="This is ground truth.",
        )

        result_data = self._extract_and_print_result(results, "response-as-non-string")
        self.assert_pass_or_fail(result_data)

    def test_ground_truth_as_non_string_type(self) -> None:
        """Test case: Ground truth provided as non-string type.

        Tests handling when ground_truth is not a string. Currently expects graceful handling
        (e.g., automatic string conversion).

        TODO: Decide if this should raise an error instead.
        """
        results = self._run_evaluation(
            response="This is a response.",
            ground_truth=12345,  # Integer instead of string
        )

        result_data = self._extract_and_print_result(results, "ground-truth-as-non-string")
        self.assert_pass_or_fail(result_data)

    def test_response_as_list_type(self) -> None:
        """Test case: Response provided as list.

        Tests handling when response is a list (conversation format not supported).
        Currently expects graceful handling (e.g., string conversion or processing).

        TODO: Decide if this should raise an error for unsupported type.
        """
        results = self._run_evaluation(
            response=[{"role": "assistant", "content": "text"}],
            ground_truth="This is ground truth.",
        )

        result_data = self._extract_and_print_result(results, "response-as-list")
        self.assert_pass_or_fail(result_data)

    def test_ground_truth_as_list_type(self) -> None:
        """Test case: Ground truth provided as list.

        Tests handling when ground_truth is a list (not supported for this evaluator).
        Currently expects graceful handling (e.g., string conversion or processing).

        TODO: Decide if this should raise an error for unsupported type.
        """
        results = self._run_evaluation(
            response="This is a response.",
            ground_truth=["item1", "item2"],
        )

        result_data = self._extract_and_print_result(results, "ground-truth-as-list")
        self.assert_pass_or_fail(result_data)

    # ==================== INTERMEDIATE RESPONSE / PREPROCESSING TESTS ====================

    def test_function_call_response(self):
        """Function call types: intermediate returns not-applicable, full response is preprocessed and passes."""
        function_call_only = [
            {
                "run_id": "",
                "role": "assistant",
                "content": [
                    {
                        "type": "function_call",
                        "tool_call_id": "call_15sVz7lMj1JbY4ea0Om8oigT",
                        "name": "get_horoscope",
                        "arguments": {"sign": "Aquarius"},
                    }
                ],
            }
        ]
        function_call_full = [
            {
                "run_id": "",
                "role": "assistant",
                "content": [
                    {
                        "type": "function_call",
                        "tool_call_id": "call_15sVz7lMj1JbY4ea0Om8oigT",
                        "name": "get_horoscope",
                        "arguments": {"sign": "Aquarius"},
                    }
                ],
            },
            {
                "run_id": "",
                "tool_call_id": "call_15sVz7lMj1JbY4ea0Om8oigT",
                "role": "tool",
                "content": [
                    {
                        "type": "function_call_output",
                        "function_call_output": {
                            "horoscope": "Aquarius: Next Tuesday you will befriend a baby otter."
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "annotations": [],
                        "text": "Your horoscope for today as an Aquarius: "
                        "Next Tuesday you will befriend a baby otter.",
                        "type": "output_text",
                        "logprobs": [],
                    }
                ],
            },
        ]

        # Intermediate: function_call-only response -> not applicable
        results = self._run_evaluation(
            response=function_call_only,
            ground_truth="The horoscope for Aquarius.",
        )
        result_data = self._extract_and_print_result(results, "Function Call Only - Not Applicable")
        self.assert_not_applicable(result_data)

        # Full: function_call/function_call_output types preprocessed -> pass
        results = self._run_evaluation(
            response=function_call_full,
            ground_truth="The horoscope for Aquarius.",
        )
        result_data = self._extract_and_print_result(results, "Function Call Full - Preprocessed")
        self.assert_pass(result_data)

    def test_mcp_approval_response(self):
        """MCP approval types: intermediate returns not-applicable, full response is preprocessed and passes."""
        mcp_only = [
            {
                "run_id": "",
                "role": "assistant",
                "content": [
                    {
                        "type": "mcp_approval_request",
                        "tool_call_id": "mcpr_04f33cbf84783da400695a7330ed4c8190b37cc43c1ef54642",
                        "name": "microsoft_docs_search",
                        "arguments": {"query": "how Azure Functions work"},
                    }
                ],
            }
        ]
        mcp_full = [
            {
                "run_id": "",
                "role": "assistant",
                "content": [
                    {
                        "type": "mcp_approval_request",
                        "tool_call_id": "mcpr_04f33cbf84783da400695a7330ed4c8190b37cc43c1ef54642",
                        "name": "microsoft_docs_search",
                        "arguments": {"query": "how Azure Functions work"},
                    }
                ],
            },
            {
                "run_id": "",
                "tool_call_id": "mcpr_04f33cbf84783da400695a7330ed4c8190b37cc43c1ef54642",
                "role": "tool",
                "content": [
                    {
                        "type": "mcp_approval_response",
                        "mcp_approval_response": True,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_1",
                        "name": "microsoft_docs_search",
                        "arguments": {"query": "how Azure Functions work"},
                    }
                ],
            },
            {
                "tool_call_id": "call_1",
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": {
                            "title": "Azure Functions overview",
                            "url": "https://learn.microsoft.com/azure/azure-functions/",
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Azure Functions is a serverless compute service "
                        "that lets you run event-triggered code.",
                    }
                ],
            },
        ]

        # Intermediate: mcp_approval_request-only response -> not applicable
        results = self._run_evaluation(
            response=mcp_only,
            ground_truth="Azure Functions overview.",
        )
        result_data = self._extract_and_print_result(results, "MCP Approval Only - Not Applicable")
        self.assert_not_applicable(result_data)

        # Full: mcp_approval messages dropped, remaining evaluated -> pass
        results = self._run_evaluation(
            response=mcp_full,
            ground_truth="Azure Functions overview.",
        )
        result_data = self._extract_and_print_result(results, "MCP Approval Full - Preprocessed")
        self.assert_pass(result_data)
