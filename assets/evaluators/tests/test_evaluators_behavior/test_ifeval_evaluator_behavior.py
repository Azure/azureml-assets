# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for IFEval Evaluator."""

import asyncio
import sys
import types

import pytest
from typing import Any, Dict
from ..common.base_code_evaluator_runner import BaseCodeEvaluatorRunner
from ...builtin.ifeval.evaluator._ifeval import IFEvalEvaluator
from ...builtin.ifeval.evaluator import _instructions_util as ifeval_util
from ...builtin.ifeval.evaluator import _instructions as ifeval_instructions
from ...builtin.ifeval.evaluator._instructions import (
    InstructionChecker,
    get_checker,
)


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
        """Return the expected result fields for IFEval evaluator."""
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


@pytest.mark.unittest
class TestIFEvalInstructionCheckers:
    """White-box unit tests for the ported IFEval instruction checkers.

    These exercise each checker's pass and fail branches (and the relational
    variants and loose-tolerance overrides) directly through ``get_checker`` so
    that the instruction-following logic in ``_instructions.py`` is fully
    covered independently of the evaluator plumbing.
    """

    @staticmethod
    def _install_fake_langdetect(monkeypatch, detect_result="en", raises=False):
        """Install a fake ``langdetect`` module so language checks are deterministic."""
        fake = types.ModuleType("langdetect")

        def _detect(value):
            if raises:
                raise RuntimeError("detection failed")
            return detect_result

        fake.detect = _detect
        monkeypatch.setitem(sys.modules, "langdetect", fake)

    # region base class + registry

    def test_base_checker_not_implemented(self):
        """The abstract base raises for check_following."""
        with pytest.raises(NotImplementedError):
            InstructionChecker().check_following("anything")

    def test_get_checker_unknown_returns_none(self):
        """Unknown instruction IDs resolve to None."""
        assert get_checker("does_not:exist", {}) is None

    def test_get_checker_known_returns_instance(self):
        """Known instruction IDs resolve to a checker instance."""
        assert isinstance(get_checker("punctuation:no_comma", {}), InstructionChecker)

    def test_default_loose_delegates_to_strict(self):
        """A checker without a loose override delegates to check_following."""
        checker = get_checker("punctuation:no_comma", {})
        assert checker.check_following_loose("no commas here") is True

    def test_absolute_import_fallback(self, monkeypatch):
        """Loading the module without a package context uses the absolute-import fallback.

        The relative ``from . import _instructions_util`` fails when the module is
        loaded as a top-level module, exercising the ``except ImportError`` branch.
        """
        import importlib.util
        import os

        evaluator_dir = os.path.dirname(ifeval_instructions.__file__)
        monkeypatch.syspath_prepend(evaluator_dir)
        spec = importlib.util.spec_from_file_location(
            "_ifeval_instructions_standalone",
            os.path.join(evaluator_dir, "_instructions.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.delitem(sys.modules, "_instructions_util", raising=False)
        assert hasattr(module, "get_checker")

    # endregion

    # region language checkers (fake langdetect)

    def test_response_language_match(self, monkeypatch):
        """Detected language equal to target passes."""
        self._install_fake_langdetect(monkeypatch, "en")
        checker = get_checker("language:response_language", {"language": "en"})
        assert checker.check_following("Hello world") is True

    def test_response_language_mismatch(self, monkeypatch):
        """Detected language different from target fails."""
        self._install_fake_langdetect(monkeypatch, "fr")
        checker = get_checker("language:response_language", {"language": "en"})
        assert checker.check_following("Hello world") is False

    def test_response_language_detection_failure(self, monkeypatch):
        """Detection failure counts as followed (returns True)."""
        self._install_fake_langdetect(monkeypatch, raises=True)
        checker = get_checker("language:response_language", {"language": "en"})
        assert checker.check_following("Hello world") is True

    def test_capital_english_not_upper(self):
        """Non-uppercase text fails immediately."""
        checker = get_checker("change_case:english_capital", {})
        assert checker.check_following("lowercase text") is False

    def test_capital_english_match(self, monkeypatch):
        """Uppercase English text passes."""
        self._install_fake_langdetect(monkeypatch, "en")
        checker = get_checker("change_case:english_capital", {})
        assert checker.check_following("HELLO WORLD") is True

    def test_capital_english_mismatch(self, monkeypatch):
        """Uppercase but non-English fails."""
        self._install_fake_langdetect(monkeypatch, "fr")
        checker = get_checker("change_case:english_capital", {})
        assert checker.check_following("HELLO WORLD") is False

    def test_capital_english_detection_failure(self, monkeypatch):
        """Detection failure on uppercase text passes."""
        self._install_fake_langdetect(monkeypatch, raises=True)
        checker = get_checker("change_case:english_capital", {})
        assert checker.check_following("HELLO WORLD") is True

    def test_lowercase_english_not_lower(self):
        """Non-lowercase text fails immediately."""
        checker = get_checker("change_case:english_lowercase", {})
        assert checker.check_following("UPPERCASE TEXT") is False

    def test_lowercase_english_match(self, monkeypatch):
        """Lowercase English text passes."""
        self._install_fake_langdetect(monkeypatch, "en")
        checker = get_checker("change_case:english_lowercase", {})
        assert checker.check_following("hello world") is True

    def test_lowercase_english_mismatch(self, monkeypatch):
        """Lowercase but non-English fails."""
        self._install_fake_langdetect(monkeypatch, "fr")
        checker = get_checker("change_case:english_lowercase", {})
        assert checker.check_following("hello world") is False

    def test_lowercase_english_detection_failure(self, monkeypatch):
        """Detection failure on lowercase text passes."""
        self._install_fake_langdetect(monkeypatch, raises=True)
        checker = get_checker("change_case:english_lowercase", {})
        assert checker.check_following("hello world") is True

    # endregion

    # region length constraints

    def test_number_of_sentences_at_least(self):
        """'at least' relation on sentence count."""
        checker = get_checker(
            "length_constraints:number_sentences",
            {"num_sentences": 2, "relation": "at least"},
        )
        assert checker.check_following("First sentence. Second sentence.") is True
        assert checker.check_following("Only one sentence.") is False

    def test_number_of_sentences_less_than(self):
        """'less than' relation on sentence count."""
        checker = get_checker(
            "length_constraints:number_sentences",
            {"num_sentences": 2, "relation": "less than"},
        )
        assert checker.check_following("Only one sentence.") is True
        assert checker.check_following("First. Second. Third.") is False

    def test_number_of_sentences_loose(self):
        """Loose tolerance allows off-by-one on both relations."""
        at_least = get_checker(
            "length_constraints:number_sentences",
            {"num_sentences": 3, "relation": "at least"},
        )
        assert at_least.check_following_loose("First. Second.") is True
        less_than = get_checker(
            "length_constraints:number_sentences",
            {"num_sentences": 2, "relation": "less than"},
        )
        assert less_than.check_following_loose("First. Second.") is True

    def test_number_of_words_at_least(self):
        """'at least' relation on word count."""
        checker = get_checker(
            "length_constraints:number_words",
            {"num_words": 5, "relation": "at least"},
        )
        assert checker.check_following("one two three four five") is True
        assert checker.check_following("one two three") is False

    def test_number_of_words_less_than(self):
        """'less than' relation on word count."""
        checker = get_checker(
            "length_constraints:number_words",
            {"num_words": 5, "relation": "less than"},
        )
        assert checker.check_following("one two three") is True
        assert checker.check_following("one two three four five six") is False

    def test_number_of_words_loose(self):
        """Loose tolerance applies a percentage margin on both relations."""
        at_least = get_checker(
            "length_constraints:number_words",
            {"num_words": 10, "relation": "at least"},
        )
        assert at_least.check_following_loose("one two three four five") is True
        less_than = get_checker(
            "length_constraints:number_words",
            {"num_words": 5, "relation": "less than"},
        )
        assert less_than.check_following_loose("one two three four five six seven") is True

    def test_paragraph_first_word_pass(self):
        """Correct paragraph count and first word passes."""
        checker = get_checker(
            "length_constraints:nth_paragraph_first_word",
            {"num_paragraphs": 2, "nth_paragraph": 1, "first_word": "hello"},
        )
        assert checker.check_following("Hello world here.\n\nSecond paragraph text.") is True

    def test_paragraph_first_word_count_mismatch(self):
        """Wrong paragraph count fails."""
        checker = get_checker(
            "length_constraints:nth_paragraph_first_word",
            {"num_paragraphs": 3, "nth_paragraph": 1, "first_word": "hello"},
        )
        assert checker.check_following("Hello world.\n\nSecond.") is False

    def test_paragraph_first_word_wrong_word(self):
        """Correct count but wrong first word fails."""
        checker = get_checker(
            "length_constraints:nth_paragraph_first_word",
            {"num_paragraphs": 2, "nth_paragraph": 1, "first_word": "goodbye"},
        )
        assert checker.check_following("Hello world.\n\nSecond para.") is False

    def test_paragraph_first_word_nth_out_of_range(self):
        """nth_paragraph beyond the paragraph list fails."""
        checker = get_checker(
            "length_constraints:nth_paragraph_first_word",
            {"num_paragraphs": 1, "nth_paragraph": 5, "first_word": "hello"},
        )
        assert checker.check_following("Only one paragraph here.") is False

    def test_paragraph_first_word_empty_nth(self):
        """An empty nth paragraph fails."""
        checker = get_checker(
            "length_constraints:nth_paragraph_first_word",
            {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "x"},
        )
        assert checker.check_following("First para.\n\n\n\nThird para.") is False

    # endregion

    # region detectable content / format

    def test_placeholder_checker(self):
        """Bracketed placeholders are counted."""
        checker = get_checker("detectable_content:number_placeholders", {"num_placeholders": 2})
        assert checker.check_following("Hello [NAME] at [PLACE]") is True
        assert checker.check_following("No placeholders") is False

    def test_bullet_list_checker(self):
        """Asterisk bullets are counted exactly."""
        checker = get_checker("detectable_format:number_bullet_lists", {"num_bullets": 3})
        assert checker.check_following("* a\n* b\n* c") is True
        assert checker.check_following("* a") is False

    def test_bullet_list_checker_loose_with_dashes(self):
        """Dash bullets count and loose allows off-by-one."""
        checker = get_checker("detectable_format:number_bullet_lists", {"num_bullets": 3})
        assert checker.check_following_loose("- a\n- b") is True

    def test_constrained_response_checker(self):
        """One of the fixed answer options must be present."""
        checker = get_checker("detectable_format:constrained_response", {})
        assert checker.check_following("My answer is yes.") is True
        assert checker.check_following("Something else") is False

    def test_highlight_section_checker_single(self):
        """Single-asterisk highlights are counted."""
        checker = get_checker("detectable_format:number_highlighted_sections", {"num_highlights": 2})
        assert checker.check_following("*first* and *second*") is True
        assert checker.check_following("no highlights") is False

    def test_highlight_section_checker_double(self):
        """Double-asterisk highlights are counted."""
        checker = get_checker("detectable_format:number_highlighted_sections", {"num_highlights": 1})
        assert checker.check_following("**bold**") is True

    def test_section_checker(self):
        """Numbered sections are split and counted."""
        checker = get_checker(
            "detectable_format:multiple_sections",
            {"section_spliter": "Section", "num_sections": 2},
        )
        assert checker.check_following("Section 1 intro Section 2 body") is True
        assert checker.check_following("no sections here") is False

    def test_paragraph_checker(self):
        """Triple-asterisk paragraph dividers are counted."""
        checker = get_checker("detectable_format:number_paragraphs", {"num_paragraphs": 2})
        assert checker.check_following("First paragraph.\n***\nSecond paragraph.") is True

    def test_paragraph_checker_empty_middle_fails(self):
        """An empty middle paragraph fails."""
        checker = get_checker("detectable_format:number_paragraphs", {"num_paragraphs": 2})
        assert checker.check_following("First\n***\n***\nSecond") is False

    def test_paragraph_checker_trailing_empty(self):
        """A trailing divider produces an empty final paragraph that is not counted."""
        checker = get_checker("detectable_format:number_paragraphs", {"num_paragraphs": 2})
        assert checker.check_following("First para\n***\nSecond para\n***") is True

    def test_postscript_checker_ps(self):
        """A P.S. marker is detected."""
        checker = get_checker("detectable_content:postscript", {"postscript_marker": "P.S."})
        assert checker.check_following("Main text. P.S. Extra note") is True
        assert checker.check_following("No postscript") is False

    def test_postscript_checker_pps(self):
        """A P.P.S marker is detected."""
        checker = get_checker("detectable_content:postscript", {"postscript_marker": "P.P.S"})
        assert checker.check_following("Body. P.P.S. more") is True

    def test_postscript_checker_custom_marker(self):
        """A custom postscript marker is detected."""
        checker = get_checker("detectable_content:postscript", {"postscript_marker": "NOTE:"})
        assert checker.check_following("text NOTE: hi") is True

    def test_json_format(self):
        """Valid JSON passes, invalid text fails."""
        checker = get_checker("detectable_format:json_format", {})
        assert checker.check_following('{"key": "value"}') is True
        assert checker.check_following("not json at all") is False

    def test_json_format_with_markdown_fence(self):
        """JSON wrapped in a markdown fence is unwrapped and validated."""
        checker = get_checker("detectable_format:json_format", {})
        assert checker.check_following('```json\n{"a": 1}\n```') is True

    def test_title_checker(self):
        """A non-empty title in double angle brackets passes."""
        checker = get_checker("detectable_format:title", {})
        assert checker.check_following("<<My Title>>\nContent") is True
        assert checker.check_following("No title here") is False

    def test_title_checker_blank_title(self):
        """A whitespace-only title is not counted."""
        checker = get_checker("detectable_format:title", {})
        assert checker.check_following("<<   >>") is False

    # endregion

    # region keywords

    def test_keyword_checker(self):
        """All required keywords must be present."""
        checker = get_checker("keywords:existence", {"keywords": ["fox", "dog"]})
        assert checker.check_following("The fox and the dog") is True
        assert checker.check_following("Only the fox") is False

    def test_keyword_frequency_at_least(self):
        """'at least' relation on keyword frequency."""
        checker = get_checker(
            "keywords:frequency",
            {"keyword": "the", "frequency": 2, "relation": "at least"},
        )
        assert checker.check_following("the cat and the dog") is True
        assert checker.check_following("the cat") is False

    def test_keyword_frequency_less_than(self):
        """'less than' relation on keyword frequency."""
        checker = get_checker(
            "keywords:frequency",
            {"keyword": "the", "frequency": 2, "relation": "less than"},
        )
        assert checker.check_following("the cat") is True
        assert checker.check_following("the the the") is False

    def test_forbidden_words(self):
        """Forbidden words must not appear."""
        checker = get_checker("keywords:forbidden_words", {"forbidden_words": ["bad"]})
        assert checker.check_following("all good here") is True
        assert checker.check_following("this is bad") is False

    def test_letter_frequency_at_least(self):
        """'at least' relation on letter frequency."""
        checker = get_checker(
            "keywords:letter_frequency",
            {"letter": "a", "let_frequency": 3, "let_relation": "at least"},
        )
        assert checker.check_following("banana aardvark") is True
        assert checker.check_following("hello") is False

    def test_letter_frequency_less_than(self):
        """'less than' relation on letter frequency."""
        checker = get_checker(
            "keywords:letter_frequency",
            {"letter": "z", "let_frequency": 1, "let_relation": "less than"},
        )
        assert checker.check_following("no zed here") is False
        assert checker.check_following("hello") is True

    # endregion

    # region change case / punctuation / startend

    def test_comma_checker(self):
        """No commas allowed."""
        checker = get_checker("punctuation:no_comma", {})
        assert checker.check_following("no commas here") is True
        assert checker.check_following("has, a comma") is False

    def test_capital_word_frequency_at_least(self):
        """'at least' relation on all-caps word frequency."""
        checker = get_checker(
            "change_case:capital_word_frequency",
            {"capital_frequency": 2, "capital_relation": "at least"},
        )
        assert checker.check_following("HELLO WORLD lowercase") is True
        assert checker.check_following("only ONE capital") is False

    def test_capital_word_frequency_less_than(self):
        """'less than' relation on all-caps word frequency."""
        checker = get_checker(
            "change_case:capital_word_frequency",
            {"capital_frequency": 2, "capital_relation": "less than"},
        )
        assert checker.check_following("only ONE") is True
        assert checker.check_following("TWO BIG WORDS") is False

    def test_end_checker(self):
        """Response must end with the required phrase."""
        checker = get_checker("startend:end_checker", {"end_phrase": "the end"})
        assert checker.check_following("It is the end") is True
        assert checker.check_following("not matching") is False

    def test_end_checker_empty_phrase(self):
        """An empty end phrase always passes."""
        checker = get_checker("startend:end_checker", {})
        assert checker.check_following("anything") is True

    def test_constrained_start_checker(self):
        """Response must start with the required phrase."""
        checker = get_checker("startend:constrained_start", {"starter": "First"})
        assert checker.check_following("First, let me explain") is True
        assert checker.check_following("Let me explain now") is False

    def test_constrained_start_empty(self):
        """An empty starter always passes."""
        checker = get_checker("startend:constrained_start", {})
        assert checker.check_following("anything") is True

    def test_quotation_checker(self):
        """Response must be wrapped in double quotes."""
        checker = get_checker("startend:quotation", {})
        assert checker.check_following('"quoted text"') is True
        assert checker.check_following("unquoted") is False

    def test_quotation_checker_too_short(self):
        """A single character cannot be wrapped in quotes."""
        checker = get_checker("startend:quotation", {})
        assert checker.check_following('"') is False

    # endregion

    # region combination

    def test_two_responses_checker(self):
        """Two distinct responses separated by the divider pass."""
        checker = get_checker("combination:two_responses", {})
        assert checker.check_following("First answer\n******\nSecond answer") is True
        assert checker.check_following("Only one answer") is False

    def test_two_responses_identical_fails(self):
        """Two identical responses fail."""
        checker = get_checker("combination:two_responses", {})
        assert checker.check_following("same\n******\nsame") is False

    def test_two_responses_empty_middle_fails(self):
        """An empty middle segment fails."""
        checker = get_checker("combination:two_responses", {})
        assert checker.check_following("a\n******\n******\nb") is False

    def test_repeat_prompt_checker(self):
        """Response must begin with the repeated prompt."""
        checker = get_checker("combination:repeat_prompt", {"prompt_to_repeat": "Repeat this"})
        assert checker.check_following("Repeat this and then answer") is True
        assert checker.check_following("Different start") is False

    def test_repeat_prompt_empty(self):
        """An empty prompt to repeat always passes."""
        checker = get_checker("combination:repeat_prompt", {})
        assert checker.check_following("anything") is True

    # endregion


@pytest.mark.unittest
class TestIFEvalDoEvalBranches:
    """White-box tests for IFEvalEvaluator._do_eval and _parse_json_field.

    Exercises the JSON parsing helper and the validation / error branches of
    ``_do_eval`` that the black-box behavioral tests do not reach.
    """

    def test_parse_json_field_none(self):
        """None input returns None."""
        assert IFEvalEvaluator._parse_json_field(None, "field") is None

    def test_parse_json_field_invalid(self):
        """Invalid JSON string returns None."""
        assert IFEvalEvaluator._parse_json_field("not json {", "field") is None

    def test_parse_json_field_already_parsed(self):
        """A non-string value is returned unchanged."""
        assert IFEvalEvaluator._parse_json_field([1, 2], "field") == [1, 2]

    def test_do_eval_empty_response(self):
        """An empty response fails."""
        result = asyncio.run(IFEvalEvaluator()._do_eval({"response": ""}))
        assert result["ifeval_strict"] is False

    def test_do_eval_empty_instruction_list(self):
        """An empty instruction id list fails."""
        result = asyncio.run(
            IFEvalEvaluator()._do_eval({"response": "hi", "instruction_id_list": "[]"})
        )
        assert result["ifeval_strict"] is False

    def test_do_eval_length_mismatch(self):
        """Mismatched instruction id / kwargs lengths fail."""
        result = asyncio.run(
            IFEvalEvaluator()._do_eval(
                {
                    "response": "hi",
                    "instruction_id_list": '["punctuation:no_comma", "startend:quotation"]',
                    "instruction_kwargs": "[{}]",
                }
            )
        )
        assert result["ifeval_strict"] is False

    def test_do_eval_unknown_instruction(self):
        """An unknown instruction id yields a failing result."""
        result = asyncio.run(
            IFEvalEvaluator()._do_eval(
                {
                    "response": "hi there",
                    "instruction_id_list": '["bogus:unknown"]',
                    "instruction_kwargs": "[{}]",
                }
            )
        )
        assert result["ifeval_strict"] is False

    def test_do_eval_checker_exception(self, monkeypatch):
        """An exception raised inside a checker is caught and treated as a failure."""

        def _boom(self, value):
            raise RuntimeError("boom")

        monkeypatch.setattr(ifeval_instructions.CommaChecker, "check_following", _boom)
        result = asyncio.run(
            IFEvalEvaluator()._do_eval(
                {
                    "response": "some text",
                    "instruction_id_list": '["punctuation:no_comma"]',
                    "instruction_kwargs": "[{}]",
                }
            )
        )
        assert result["ifeval_strict"] is False

    def test_do_eval_success(self):
        """A satisfied instruction yields a passing result."""
        result = asyncio.run(
            IFEvalEvaluator()._do_eval(
                {
                    "response": "no commas here",
                    "instruction_id_list": '["punctuation:no_comma"]',
                    "instruction_kwargs": "[{}]",
                }
            )
        )
        assert result["ifeval_strict"] is True

    def test_do_eval_missing_kwargs_defaults(self):
        """Omitting instruction_kwargs defaults to empty parameter dicts."""
        result = asyncio.run(
            IFEvalEvaluator()._do_eval(
                {
                    "response": "no commas here",
                    "instruction_id_list": '["punctuation:no_comma"]',
                }
            )
        )
        assert result["ifeval_strict"] is True


@pytest.mark.unittest
class TestIFEvalInstructionsUtil:
    """Unit tests for the sentence/word utilities in _instructions_util.py."""

    def test_split_into_sentences_basic(self):
        """Basic sentences are split on terminal punctuation."""
        sentences = ifeval_util.split_into_sentences("Hello world. This is a test. Done!")
        assert len(sentences) == 3

    def test_split_into_sentences_special_cases(self):
        """Prefixes, acronyms, websites, quotes, and Ph.D. are handled."""
        text = (
            'Dr. Smith paid 3.50 at 4.5 percent. '
            'He said "Hello!" and asked "Really?" '
            'The A.B.C. Corp and Inc. Ltd. left... '
            'Visit www.example.com today. Ph.D. holders rejoice.'
        )
        sentences = ifeval_util.split_into_sentences(text)
        assert isinstance(sentences, list)
        assert len(sentences) > 0

    def test_count_words(self):
        """Words are counted via word tokens."""
        assert ifeval_util.count_words("one two three") == 3

    def test_count_sentences(self):
        """Sentences are counted."""
        assert ifeval_util.count_sentences("First. Second.") == 2

    def test_generate_keywords(self):
        """Requested number of keywords is returned."""
        assert len(ifeval_util.generate_keywords(3)) == 3

    def test_generate_keywords_capped(self):
        """Requesting more than the word list caps at the list length."""
        assert len(ifeval_util.generate_keywords(100000)) == len(ifeval_util.WORD_LIST)
