# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for IFEval Evaluator."""

import pytest
from typing import Any, Dict
from ..common.base_code_evaluator_runner import BaseCodeEvaluatorRunner
from ...builtin.ifeval.evaluator._ifeval import IFEvalEvaluator


@pytest.mark.unittest
class TestIFEvalEvaluatorBehavior(BaseCodeEvaluatorRunner):
    """
    Behavioral tests for IFEval (Instruction-Following Evaluation) Evaluator.

    Tests the instruction checking logic ported from google-research/instruction_following_eval:
    - 25+ instruction checker types
    - Strict vs loose accuracy modes
    - Multiple instruction handling
    - JSON parameter parsing
    """

    evaluator_type = IFEvalEvaluator
    result_key = "ifeval_strict"
    result_prefix = "ifeval"

    # Override expected result fields for IFEval (has both strict and loose)
    @property
    def expected_result_fields(self):
        return [
            "ifeval_strict",
            "ifeval_loose",
            "ifeval_result",
        ]

    # IFEval returns boolean scores
    def assert_pass(self, result_data: Dict[str, Any]):
        """Assert a passing result (boolean True)."""
        assert result_data["label"] == "pass", f"Expected 'pass' but got '{result_data['label']}'"
        assert result_data["score"] is True, f"Expected True but got {result_data['score']}"

    def assert_fail(self, result_data: Dict[str, Any]):
        """Assert a failing result (boolean False)."""
        assert result_data["label"] == "fail", f"Expected 'fail' but got '{result_data['label']}'"
        assert result_data["score"] is False, f"Expected False but got {result_data['score']}"

    # region Test Data
    # No comma instruction
    NO_COMMA_RESPONSE = "This response has no commas at all"
    WITH_COMMA_RESPONSE = "Hello, world!"

    # Word count instruction
    TEN_WORD_RESPONSE = "one two three four five six seven eight nine ten eleven twelve"
    THREE_WORD_RESPONSE = "one two three"
    EIGHT_WORD_RESPONSE = "one two three four five six seven eight"

    # JSON format
    VALID_JSON = '{"name": "test", "value": 42}'
    INVALID_JSON = "This is not JSON"
    JSON_WITH_MARKDOWN = '```json\n{"name": "test"}\n```'

    # Bullet list
    ASTERISK_BULLETS = "* First item\n* Second item\n* Third item"
    DASH_BULLETS = "- First item\n- Second item\n- Third item"

    # Keywords
    KEYWORDS_PRESENT = "The quick brown fox jumps over the lazy dog"
    KEYWORDS_MISSING = "Hello world"

    # Placeholders
    PLACEHOLDERS_PRESENT = "Hello [NAME], welcome to [PLACE]!"
    PLACEHOLDERS_MISSING = "Hello friend, welcome here!"

    # Title
    TITLED_RESPONSE = "# My Title\n\nThis is the content."
    UNTITLED_RESPONSE = "This response has no title."

    # Quoted response
    QUOTED_RESPONSE = '"This is a quoted response."'
    UNQUOTED_RESPONSE = "This is not quoted."

    # Highlights
    HIGHLIGHTED_RESPONSE = "*Important* and *critical* points here."

    # Empty
    EMPTY_STRING = ""
    # endregion

    # ==================== SINGLE INSTRUCTION TESTS ====================

    def test_no_comma_pass(self):
        """Test punctuation:no_comma instruction passes."""
        results = self._run_evaluation(
            response=self.NO_COMMA_RESPONSE,
            instruction_id_list='["punctuation:no_comma"]',
            instruction_kwargs='[{}]',
        )
        result_data = self._extract_and_print_result(results, "no_comma_pass")
        self.assert_pass(result_data)

    def test_no_comma_fail(self):
        """Test punctuation:no_comma instruction fails."""
        results = self._run_evaluation(
            response=self.WITH_COMMA_RESPONSE,
            instruction_id_list='["punctuation:no_comma"]',
            instruction_kwargs='[{}]',
        )
        result_data = self._extract_and_print_result(results, "no_comma_fail")
        self.assert_fail(result_data)

    # ==================== WORD COUNT TESTS ====================

    def test_word_count_at_least_pass(self):
        """Test word count 'at least' constraint passes."""
        results = self._run_evaluation(
            response=self.TEN_WORD_RESPONSE,
            instruction_id_list='["length_constraints:number_words"]',
            instruction_kwargs='[{"num_words": 10, "relation": "at least"}]',
        )
        result_data = self._extract_and_print_result(results, "word_count_at_least_pass")
        self.assert_pass(result_data)

    def test_word_count_at_least_fail(self):
        """Test word count 'at least' constraint fails."""
        results = self._run_evaluation(
            response=self.THREE_WORD_RESPONSE,
            instruction_id_list='["length_constraints:number_words"]',
            instruction_kwargs='[{"num_words": 10, "relation": "at least"}]',
        )
        result_data = self._extract_and_print_result(results, "word_count_at_least_fail")
        self.assert_fail(result_data)

    def test_word_count_less_than_pass(self):
        """Test word count 'less than' constraint passes."""
        results = self._run_evaluation(
            response=self.THREE_WORD_RESPONSE,
            instruction_id_list='["length_constraints:number_words"]',
            instruction_kwargs='[{"num_words": 10, "relation": "less than"}]',
        )
        result_data = self._extract_and_print_result(results, "word_count_less_than_pass")
        self.assert_pass(result_data)

    # ==================== JSON FORMAT TESTS ====================

    def test_json_format_pass(self):
        """Test JSON format instruction passes."""
        results = self._run_evaluation(
            response=self.VALID_JSON,
            instruction_id_list='["detectable_format:json_format"]',
            instruction_kwargs='[{}]',
        )
        result_data = self._extract_and_print_result(results, "json_format_pass")
        self.assert_pass(result_data)

    def test_json_format_fail(self):
        """Test JSON format instruction fails."""
        results = self._run_evaluation(
            response=self.INVALID_JSON,
            instruction_id_list='["detectable_format:json_format"]',
            instruction_kwargs='[{}]',
        )
        result_data = self._extract_and_print_result(results, "json_format_fail")
        self.assert_fail(result_data)

    def test_json_format_with_markdown(self):
        """Test JSON format with markdown code block passes."""
        results = self._run_evaluation(
            response=self.JSON_WITH_MARKDOWN,
            instruction_id_list='["detectable_format:json_format"]',
            instruction_kwargs='[{}]',
        )
        result_data = self._extract_and_print_result(results, "json_format_markdown")
        self.assert_pass(result_data)

    # ==================== BULLET LIST TESTS ====================

    def test_bullet_list_asterisk(self):
        """Test bullet list with asterisks."""
        results = self._run_evaluation(
            response=self.ASTERISK_BULLETS,
            instruction_id_list='["detectable_format:number_bullet_lists"]',
            instruction_kwargs='[{"num_bullets": 3}]',
        )
        result_data = self._extract_and_print_result(results, "bullet_list_asterisk")
        self.assert_pass(result_data)

    def test_bullet_list_dash(self):
        """Test bullet list with dashes."""
        results = self._run_evaluation(
            response=self.DASH_BULLETS,
            instruction_id_list='["detectable_format:number_bullet_lists"]',
            instruction_kwargs='[{"num_bullets": 3}]',
        )
        result_data = self._extract_and_print_result(results, "bullet_list_dash")
        self.assert_pass(result_data)

    # ==================== KEYWORD TESTS ====================

    def test_keywords_present(self):
        """Test keywords existence instruction passes."""
        results = self._run_evaluation(
            response=self.KEYWORDS_PRESENT,
            instruction_id_list='["keywords:existence"]',
            instruction_kwargs='[{"keywords": ["quick", "fox", "dog"]}]',
        )
        result_data = self._extract_and_print_result(results, "keywords_present")
        self.assert_pass(result_data)

    def test_keywords_missing(self):
        """Test keywords existence instruction fails."""
        results = self._run_evaluation(
            response=self.KEYWORDS_MISSING,
            instruction_id_list='["keywords:existence"]',
            instruction_kwargs='[{"keywords": ["quick", "fox", "dog"]}]',
        )
        result_data = self._extract_and_print_result(results, "keywords_missing")
        self.assert_fail(result_data)

    # ==================== MULTIPLE INSTRUCTION TESTS ====================

    def test_multiple_instructions_all_pass(self):
        """Test all instructions passing."""
        results = self._run_evaluation(
            response=self.TEN_WORD_RESPONSE,
            instruction_id_list='["punctuation:no_comma", "length_constraints:number_words"]',
            instruction_kwargs='[{}, {"num_words": 10, "relation": "at least"}]',
        )
        result_data = self._extract_and_print_result(results, "multiple_all_pass")
        self.assert_pass(result_data)

    def test_multiple_instructions_one_fails(self):
        """Test failure when one instruction fails."""
        # Has commas, so no_comma fails
        results = self._run_evaluation(
            response="one, two, three",
            instruction_id_list='["punctuation:no_comma", "length_constraints:number_words"]',
            instruction_kwargs='[{}, {"num_words": 3, "relation": "at least"}]',
        )
        result_data = self._extract_and_print_result(results, "multiple_one_fails")
        self.assert_fail(result_data)

    # ==================== STRICT VS LOOSE TESTS ====================

    def test_strict_fail_loose_pass(self):
        """Test case where strict fails but loose passes (10% tolerance)."""
        # Word count requirement is 10, response has 8 words
        # Strict should fail, loose should pass
        results = self._run_evaluation(
            response=self.EIGHT_WORD_RESPONSE,
            instruction_id_list='["length_constraints:number_words"]',
            instruction_kwargs='[{"num_words": 10, "relation": "at least"}]',
        )
        # Check raw results for both strict and loose
        assert results.get("ifeval_strict") is False, "Strict should fail"
        assert results.get("ifeval_loose") is True, "Loose should pass with tolerance"

    # ==================== EDGE CASE TESTS ====================

    def test_empty_response(self):
        """Test empty response returns fail."""
        results = self._run_evaluation(
            response=self.EMPTY_STRING,
            instruction_id_list='["punctuation:no_comma"]',
            instruction_kwargs='[{}]',
        )
        result_data = self._extract_and_print_result(results, "empty_response")
        self.assert_fail(result_data)

    def test_empty_instruction_list(self):
        """Test empty instruction list returns fail."""
        results = self._run_evaluation(
            response="Some response",
            instruction_id_list='[]',
            instruction_kwargs='[]',
        )
        result_data = self._extract_and_print_result(results, "empty_instruction_list")
        self.assert_fail(result_data)
