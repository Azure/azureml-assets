# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Huggingface predict file for whisper mlflow model."""

import os
import re
import base64
import ffmpeg
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict
from transformers import WhisperProcessor, WhisperForConditionalGeneration


SUPPORTED_LANGUAGES = [
    "en",
    "zh",
    "de",
    "es",
    "ru",
    "ko",
    "fr",
    "ja",
    "pt",
    "tr",
    "pl",
    "ca",
    "nl",
    "ar",
    "sv",
    "it",
    "id",
    "hi",
    "fi",
    "vi",
    "he",
    "uk",
    "el",
    "ms",
    "cs",
    "ro",
    "da",
    "hu",
    "ta",
    "'no'",
    "th",
    "ur",
    "hr",
    "bg",
    "lt",
    "la",
    "mi",
    "ml",
    "cy",
    "sk",
    "te",
    "fa",
    "lv",
    "bn",
    "sr",
    "az",
    "sl",
    "kn",
    "et",
    "mk",
    "br",
    "eu",
    "is",
    "hy",
    "ne",
    "mn",
    "bs",
    "kk",
    "sq",
    "sw",
    "gl",
    "mr",
    "pa",
    "si",
    "km",
    "sn",
    "yo",
    "so",
    "af",
    "oc",
    "ka",
    "be",
    "tg",
    "sd",
    "gu",
    "am",
    "yi",
    "lo",
    "uz",
    "fo",
    "ht",
    "ps",
    "tk",
    "nn",
    "mt",
    "sa",
    "lb",
    "my",
    "bo",
    "tl",
    "mg",
    "as",
    "tt",
    "haw",
    "ln",
    "ha",
    "ba",
    "jw",
    "su",
]


def is_valid_url(str):
    """Return if URL is a valid string."""
    regex = (
        "((http|https)://)(www.)?"
        + "[a-zA-Z0-9@:%._\\+~#?&//=]"
        + "{2,256}\\.[a-z]"
        + "{2,6}\\b([-a-zA-Z0-9@:%"
        + "._\\+~#?&//=]*)"
    )
    p = re.compile(regex)
    # If the string is empty
    # return false
    if str is None:
        return False
    # Return if the string
    # matched the ReGex
    if re.search(p, str):
        return True
    else:
        return False


def audio_input_to_nparray(audio_file_path: Path, sampling_rate: int = 16000) -> np.array:
    """Convert base64encoded audio string to np array."""
    try:
        out, _ = (
            ffmpeg.input(audio_file_path, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sampling_rate)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        return f"Failed to load audio: {e.stderr.decode()}"

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def audio_processor(audio_input: str, sampling_rate: int = 16000) -> np.array:
    """Process input audio and convert it to np array."""
    with TemporaryDirectory() as temp_dir:
        audio_file_path = os.path.join(temp_dir, "audio_file.m4a")

        # check if input string is a URI to an audio file
        if is_valid_url(audio_input):
            with open(audio_file_path, "wb") as f:
                f.write(requests.get(audio_input).content)
            return audio_input_to_nparray(audio_file_path, sampling_rate)

        try:
            with open(audio_file_path, "wb") as audio:
                audio.write(base64.b64decode(audio_input))
        except Exception:
            raise ValueError("Invalid audio format")
        return audio_input_to_nparray(audio_file_path, sampling_rate)


def predict(
    model_input: pd.DataFrame,
    task: str,
    model: WhisperForConditionalGeneration,
    tokenizer: WhisperProcessor,
    **kwargs: Dict,
) -> str:
    """Return a whisper predicted text from an audio input.

    :param model_input: base64 encoded audio input
    :type model_input: pd.DataFrame
    :param model: whishper model
    :type model: WhisperForConditionalGeneration
    :param tokenizer: whisper model processor
    :type tokenizer: WhisperProcessor
    :param kwargs: any other args
    :type kwargs: Dict
    :return: whisper predicted text
    :rtype: str
    """
    if not task == "automatic-speech-recognition":
        return f"Invalid task name {task}"

    result = []
    for row in model_input.itertuples():
        # Parse inputs.
        audio = row.audio
        language = row.language or None
        if not isinstance(audio, str):
            return f"Invalid input format {type(audio)}, input should be base64 encoded string"
        if not isinstance(language, str):
            return f"Invalid language format {type(language)}, should be type string"
        if language not in SUPPORTED_LANGUAGES:
            return f"Language not supported {type(language)}, should be type string"

        forced_decoder_ids = (
            tokenizer.get_decoder_prompt_ids(language=language, task="transcribe") if language else None
        )
        audio_array = audio_processor(audio)
        input_features = tokenizer(audio_array, sampling_rate=16000, return_tensors="pt").input_features
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        result.append({"text": transcription})
    return result
