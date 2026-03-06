# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Utility functions for IFEval instruction checking.

Ported from Google Research's instruction_following_eval:
https://github.com/google-research/google-research/tree/master/instruction_following_eval
"""

import functools
import random
import re
from typing import List

# Word list for generating random keywords
WORD_LIST = [
    "western", "sentence", "signal", "spot", "opposite", "bottom", "potato",
    "administration", "working", "welcome", "morning", "good", "agency",
    "primary", "wish", "responsibility", "press", "problem", "president",
    "type", "beat", "trainer", "growth", "lock", "bone", "case", "equal",
    "comfortable", "region", "replacement", "performance", "mate", "walk",
    "medicine", "film", "thing", "rock", "tap", "total", "competition",
    "south", "establishment", "gather", "world", "plenty", "breath", "claim",
    "trade", "dear", "highlight", "street", "matter", "decision", "agreement",
    "studio", "coach", "assist", "brain", "wing", "style", "private", "top",
    "brown", "leg", "buy", "procedure", "method", "speed", "high", "company",
    "valuable", "analyst", "session", "pattern", "district", "pleasure",
    "dinner", "joke", "order", "plate", "department", "motor", "cell", "spend",
    "cabinet", "difference", "power", "examination", "engine", "horse",
    "dimension", "pay", "curve", "literature", "fire", "possibility", "debate",
    "activity", "passage", "hello", "cycle", "background", "quiet", "author",
    "effect", "actor", "page", "error", "throat", "attack", "character",
    "phone", "tea", "increase", "outcome", "file", "specific", "inspector",
    "internal", "potential", "staff", "building", "employer", "shoe", "hand",
]

# ISO 639-1 codes to language names
LANGUAGE_CODES = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "ar": "Arabic",
    "hi": "Hindi",
    "fr": "French",
    "ru": "Russian",
    "de": "German",
    "ja": "Japanese",
    "it": "Italian",
    "bn": "Bengali",
    "uk": "Ukrainian",
    "th": "Thai",
    "ur": "Urdu",
    "ta": "Tamil",
    "te": "Telugu",
    "bg": "Bulgarian",
    "ko": "Korean",
    "pl": "Polish",
    "he": "Hebrew",
    "fa": "Persian",
    "vi": "Vietnamese",
    "ne": "Nepali",
    "sw": "Swahili",
    "kn": "Kannada",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ml": "Malayalam",
    "fi": "Finnish",
}

# Regex patterns for sentence splitting
_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex-based rules.

    :param text: Text containing one or more sentences.
    :type text: str
    :return: List of sentence strings.
    :rtype: List[str]
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(
        _MULTIPLE_DOTS,
        lambda match: "<prd>" * len(match.group(0)) + "<stop>",
        text,
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(
        _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )
    text = re.sub(
        _ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text
    )
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def count_words(text: str) -> int:
    """Count the number of words in text.

    :param text: The text to count words in.
    :type text: str
    :return: Number of words.
    :rtype: int
    """
    tokens = re.findall(r"\w+", text)
    return len(tokens)


def count_sentences(text: str) -> int:
    """Count the number of sentences in text.

    :param text: The text to count sentences in.
    :type text: str
    :return: Number of sentences.
    :rtype: int
    """
    sentences = split_into_sentences(text)
    return len(sentences)


def generate_keywords(num_keywords: int) -> List[str]:
    """Randomly generate keywords from the word list.

    :param num_keywords: Number of keywords to generate.
    :type num_keywords: int
    :return: List of random keywords.
    :rtype: List[str]
    """
    return random.sample(WORD_LIST, k=min(num_keywords, len(WORD_LIST)))
