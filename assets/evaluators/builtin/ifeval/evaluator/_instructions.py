# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
IFEval instruction checkers.

Ported from Google Research's instruction_following_eval:
https://github.com/google-research/google-research/tree/master/instruction_following_eval

Each instruction checker implements a `check_following` method that returns True
if the response follows the instruction, False otherwise.
"""

import collections
import json
import logging
import re
from typing import Any, Dict, Optional

# Support both relative and absolute imports for testing
try:
    from . import _instructions_util as util
except ImportError:
    import _instructions_util as util

logger = logging.getLogger(__name__)

# Relational operators for comparison
COMPARISON_RELATION = ("less than", "at least")

# Constrained response options
CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe."
)


class InstructionChecker:
    """Base class for instruction checkers."""

    def __init__(self, **kwargs):
        """Initialize with instruction arguments."""
        self.kwargs = kwargs

    def check_following(self, value: str) -> bool:
        """Check if the response follows the instruction.

        :param value: The response text to check.
        :type value: str
        :return: True if the instruction is followed, False otherwise.
        :rtype: bool
        """
        raise NotImplementedError("Subclasses must implement check_following")

    def check_following_loose(self, value: str) -> bool:
        """Check if the response follows the instruction with loose tolerance.

        Default implementation is the same as strict. Subclasses can override.

        :param value: The response text to check.
        :type value: str
        :return: True if the instruction is followed (loosely), False otherwise.
        :rtype: bool
        """
        return self.check_following(value)


class ResponseLanguageChecker(InstructionChecker):
    """Check the language of the entire response."""

    def check_following(self, value: str) -> bool:
        """Check if the response is in the expected language."""
        language = self.kwargs.get("language", "en")
        try:
            import langdetect
            detected = langdetect.detect(value)
            return detected == language
        except Exception as e:
            logger.warning("Language detection failed: %s", e)
            return True  # Count as followed if detection fails


class NumberOfSentences(InstructionChecker):
    """Check the number of sentences."""

    def check_following(self, value: str) -> bool:
        """Check if sentence count meets the requirement."""
        num_sentences_threshold = self.kwargs.get("num_sentences", 1)
        relation = self.kwargs.get("relation", "at least")
        actual = util.count_sentences(value)

        if relation == "less than":
            return actual < num_sentences_threshold
        else:  # "at least"
            return actual >= num_sentences_threshold

    def check_following_loose(self, value: str) -> bool:
        """Allow off-by-one tolerance."""
        num_sentences_threshold = self.kwargs.get("num_sentences", 1)
        relation = self.kwargs.get("relation", "at least")
        actual = util.count_sentences(value)

        if relation == "less than":
            return actual < num_sentences_threshold + 1
        else:
            return actual >= num_sentences_threshold - 1


class PlaceholderChecker(InstructionChecker):
    """Check for placeholders in template writing."""

    def check_following(self, value: str) -> bool:
        """Check if response contains required number of placeholders."""
        num_placeholders = self.kwargs.get("num_placeholders", 1)
        placeholders = re.findall(r"\[.*?\]", value)
        return len(placeholders) >= num_placeholders


class BulletListChecker(InstructionChecker):
    """Check for bullet list formatting."""

    def check_following(self, value: str) -> bool:
        """Check if response has exactly the required number of bullet points."""
        num_bullets = self.kwargs.get("num_bullets", 1)
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
        total = len(bullet_lists) + len(bullet_lists_2)
        return total == num_bullets

    def check_following_loose(self, value: str) -> bool:
        """Allow off-by-one tolerance."""
        num_bullets = self.kwargs.get("num_bullets", 1)
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
        total = len(bullet_lists) + len(bullet_lists_2)
        return abs(total - num_bullets) <= 1


class ConstrainedResponseChecker(InstructionChecker):
    """Check for constrained response options."""

    def check_following(self, value: str) -> bool:
        """Check if response contains one of the allowed options."""
        value = value.strip()
        for option in CONSTRAINED_RESPONSE_OPTIONS:
            if option in value:
                return True
        return False


class ConstrainedStartChecker(InstructionChecker):
    """Check if response starts with a specific phrase."""

    def check_following(self, value: str) -> bool:
        """Check if response starts with the required phrase."""
        starter = self.kwargs.get("starter", "")
        if not starter:
            return True
        pattern = r"^\s*" + re.escape(starter) + r".*$"
        return bool(re.search(pattern, value, flags=re.MULTILINE | re.IGNORECASE))


class HighlightSectionChecker(InstructionChecker):
    """Check for highlighted sections in markdown format."""

    def check_following(self, value: str) -> bool:
        """Check if response has required number of highlighted sections."""
        num_highlights = self.kwargs.get("num_highlights", 1)
        highlights = re.findall(r"\*[^\n\*]+\*", value)
        double_highlights = re.findall(r"\*\*[^\n\*]+\*\*", value)
        count = 0
        for h in highlights:
            if h.strip("*").strip():
                count += 1
        for h in double_highlights:
            if h.removeprefix("**").removesuffix("**").strip():
                count += 1
        return count >= num_highlights


class SectionChecker(InstructionChecker):
    """Check for section formatting."""

    def check_following(self, value: str) -> bool:
        """Check if response has required number of sections."""
        section_spliter = self.kwargs.get("section_spliter", "Section")
        num_sections = self.kwargs.get("num_sections", 1)
        pattern = r"\s?" + re.escape(section_spliter) + r"\s?\d+\s?"
        sections = re.split(pattern, value, flags=re.IGNORECASE)
        return len(sections) - 1 >= num_sections


class ParagraphChecker(InstructionChecker):
    """Check for paragraph formatting with *** dividers."""

    def check_following(self, value: str) -> bool:
        """Check if response has exactly the required number of paragraphs."""
        num_paragraphs = self.kwargs.get("num_paragraphs", 1)
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        count = len(paragraphs)
        for i, p in enumerate(paragraphs):
            if not p.strip():
                if i == 0 or i == len(paragraphs) - 1:
                    count -= 1
                else:
                    return False
        return count == num_paragraphs


class PostscriptChecker(InstructionChecker):
    """Check for postscript formatting."""

    def check_following(self, value: str) -> bool:
        """Check if response contains a postscript."""
        marker = self.kwargs.get("postscript_marker", "P.S.")
        value = value.lower()
        if marker == "P.P.S":
            pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif marker == "P.S.":
            pattern = r"\s*p\.\s?s\..*$"
        else:
            pattern = r"\s*" + re.escape(marker.lower()) + r".*$"
        return bool(re.findall(pattern, value, flags=re.MULTILINE))


class KeywordChecker(InstructionChecker):
    """Check for presence of required keywords."""

    def check_following(self, value: str) -> bool:
        """Check if all required keywords are present."""
        keywords = self.kwargs.get("keywords", [])
        for keyword in keywords:
            if not re.search(re.escape(keyword), value, flags=re.IGNORECASE):
                return False
        return True


class KeywordFrequencyChecker(InstructionChecker):
    """Check keyword frequency."""

    def check_following(self, value: str) -> bool:
        """Check if keyword appears with required frequency."""
        keyword = self.kwargs.get("keyword", "")
        frequency = self.kwargs.get("frequency", 1)
        relation = self.kwargs.get("relation", "at least")
        actual = len(re.findall(re.escape(keyword), value, flags=re.IGNORECASE))
        if relation == "less than":
            return actual < frequency
        else:
            return actual >= frequency


class NumberOfWords(InstructionChecker):
    """Check word count."""

    def check_following(self, value: str) -> bool:
        """Check if word count meets requirement."""
        num_words = self.kwargs.get("num_words", 100)
        relation = self.kwargs.get("relation", "at least")
        actual = util.count_words(value)
        if relation == "less than":
            return actual < num_words
        else:
            return actual >= num_words

    def check_following_loose(self, value: str) -> bool:
        """Allow 10% tolerance."""
        num_words = self.kwargs.get("num_words", 100)
        relation = self.kwargs.get("relation", "at least")
        actual = util.count_words(value)
        tolerance = max(5, int(num_words * 0.1))
        if relation == "less than":
            return actual < num_words + tolerance
        else:
            return actual >= num_words - tolerance


class JsonFormat(InstructionChecker):
    """Check if response is valid JSON."""

    def check_following(self, value: str) -> bool:
        """Check if response is wrapped in valid JSON."""
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
            return True
        except ValueError:
            return False


class ForbiddenWords(InstructionChecker):
    """Check that forbidden words are not used."""

    def check_following(self, value: str) -> bool:
        """Check if no forbidden words appear."""
        forbidden_words = self.kwargs.get("forbidden_words", [])
        for word in forbidden_words:
            if re.search(r"\b" + re.escape(word) + r"\b", value, flags=re.IGNORECASE):
                return False
        return True


class TwoResponsesChecker(InstructionChecker):
    """Check that two responses are given."""

    def check_following(self, value: str) -> bool:
        """Check if response contains two different answers separated by ******."""
        responses = value.split("******")
        valid = []
        for i, r in enumerate(responses):
            if not r.strip():
                if i != 0 and i != len(responses) - 1:
                    return False
            else:
                valid.append(r)
        return len(valid) == 2 and valid[0].strip() != valid[1].strip()


class RepeatPromptThenAnswer(InstructionChecker):
    """Check that prompt is repeated then answered."""

    def check_following(self, value: str) -> bool:
        """Check if response starts with the repeated prompt."""
        prompt = self.kwargs.get("prompt_to_repeat", "")
        if not prompt:
            return True
        return value.strip().lower().startswith(prompt.strip().lower())


class EndChecker(InstructionChecker):
    """Check that response ends with specific phrase."""

    def check_following(self, value: str) -> bool:
        """Check if response ends with required phrase."""
        end_phrase = self.kwargs.get("end_phrase", "")
        if not end_phrase:
            return True
        value = value.strip().strip('"').lower()
        return value.endswith(end_phrase.strip().lower())


class TitleChecker(InstructionChecker):
    """Check for title in double angular brackets."""

    def check_following(self, value: str) -> bool:
        """Check if response contains a title."""
        titles = re.findall(r"<<[^\n]+>>", value)
        for title in titles:
            if title.lstrip("<").rstrip(">").strip():
                return True
        return False


class LetterFrequencyChecker(InstructionChecker):
    """Check letter frequency."""

    def check_following(self, value: str) -> bool:
        """Check if letter appears with required frequency."""
        letter = self.kwargs.get("letter", "a").lower()
        frequency = self.kwargs.get("let_frequency", 1)
        relation = self.kwargs.get("let_relation", "at least")
        letters = collections.Counter(value.lower())
        actual = letters.get(letter, 0)
        if relation == "less than":
            return actual < frequency
        else:
            return actual >= frequency


class CapitalLettersEnglishChecker(InstructionChecker):
    """Check that response is in English and all capitals."""

    def check_following(self, value: str) -> bool:
        """Check if response is uppercase English."""
        if not value.isupper():
            return False
        try:
            import langdetect
            return langdetect.detect(value) == "en"
        except Exception:
            return True


class LowercaseLettersEnglishChecker(InstructionChecker):
    """Check that response is in English and all lowercase."""

    def check_following(self, value: str) -> bool:
        """Check if response is lowercase English."""
        if not value.islower():
            return False
        try:
            import langdetect
            return langdetect.detect(value) == "en"
        except Exception:
            return True


class CommaChecker(InstructionChecker):
    """Check that response contains no commas."""

    def check_following(self, value: str) -> bool:
        """Check if response has no commas."""
        return "," not in value


class CapitalWordFrequencyChecker(InstructionChecker):
    """Check frequency of all-caps words."""

    def check_following(self, value: str) -> bool:
        """Check if all-caps words appear with required frequency."""
        frequency = self.kwargs.get("capital_frequency", 1)
        relation = self.kwargs.get("capital_relation", "at least")
        words = re.findall(r"\b[A-Z]+\b", value)
        actual = len(words)
        if relation == "less than":
            return actual < frequency
        else:
            return actual >= frequency


class QuotationChecker(InstructionChecker):
    """Check that response is wrapped in double quotes."""

    def check_following(self, value: str) -> bool:
        """Check if response is wrapped in double quotation marks."""
        value = value.strip()
        return len(value) > 1 and value[0] == '"' and value[-1] == '"'


class ParagraphFirstWordCheck(InstructionChecker):
    """Check paragraph count and first word of nth paragraph."""

    def check_following(self, value: str) -> bool:
        """Check paragraphs and first word requirement."""
        num_paragraphs = self.kwargs.get("num_paragraphs", 1)
        nth_paragraph = self.kwargs.get("nth_paragraph", 1)
        first_word = self.kwargs.get("first_word", "").lower()

        paragraphs = re.split(r"\n\n", value)
        count = sum(1 for p in paragraphs if p.strip())

        if count != num_paragraphs:
            return False

        if nth_paragraph > len(paragraphs):
            return False

        paragraph = paragraphs[nth_paragraph - 1].strip()
        if not paragraph:
            return False

        words = paragraph.split()
        if not words:
            return False

        actual_first = re.sub(r"[^\w]", "", words[0]).lower()
        return actual_first == first_word


# Mapping from instruction IDs to checker classes
INSTRUCTION_CHECKERS: Dict[str, type] = {
    "language:response_language": ResponseLanguageChecker,
    "length_constraints:number_sentences": NumberOfSentences,
    "length_constraints:number_words": NumberOfWords,
    "detectable_content:number_placeholders": PlaceholderChecker,
    "detectable_format:number_bullet_lists": BulletListChecker,
    "detectable_format:constrained_response": ConstrainedResponseChecker,
    "startend:end_checker": EndChecker,
    "change_case:english_capital": CapitalLettersEnglishChecker,
    "change_case:english_lowercase": LowercaseLettersEnglishChecker,
    "punctuation:no_comma": CommaChecker,
    "detectable_format:number_highlighted_sections": HighlightSectionChecker,
    "detectable_format:multiple_sections": SectionChecker,
    "detectable_format:number_paragraphs": ParagraphChecker,
    "detectable_content:postscript": PostscriptChecker,
    "keywords:existence": KeywordChecker,
    "keywords:frequency": KeywordFrequencyChecker,
    "keywords:forbidden_words": ForbiddenWords,
    "keywords:letter_frequency": LetterFrequencyChecker,
    "detectable_format:json_format": JsonFormat,
    "detectable_format:title": TitleChecker,
    "combination:two_responses": TwoResponsesChecker,
    "combination:repeat_prompt": RepeatPromptThenAnswer,
    "startend:quotation": QuotationChecker,
    "length_constraints:nth_paragraph_first_word": ParagraphFirstWordCheck,
    "change_case:capital_word_frequency": CapitalWordFrequencyChecker,
    "startend:constrained_start": ConstrainedStartChecker,
}


def get_checker(instruction_id: str, kwargs: Dict[str, Any]) -> Optional[InstructionChecker]:
    """Get an instruction checker by ID.

    :param instruction_id: The instruction ID (e.g., "punctuation:no_comma").
    :type instruction_id: str
    :param kwargs: Arguments for the checker.
    :type kwargs: Dict[str, Any]
    :return: The instruction checker, or None if not found.
    :rtype: Optional[InstructionChecker]
    """
    checker_class = INSTRUCTION_CHECKERS.get(instruction_id)
    if checker_class is None:
        logger.warning("Unknown instruction ID: %s", instruction_id)
        return None
    return checker_class(**kwargs)
