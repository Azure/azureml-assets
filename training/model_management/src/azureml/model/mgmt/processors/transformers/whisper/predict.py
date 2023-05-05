# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Huggingface predict file for whisper mlflow model."""

import os
import base64
import wget
import ffmpeg
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from tempfile import TemporaryDirectory
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from urllib.parse import urlparse

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.WARNING)

supported_languages = [
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
    result = urlparse(str)
    return all([result.scheme, result.netloc])


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

        if is_valid_url(audio_input):
            logging.info("Recieved Remote Audio URL as input.")
            try:
                wget.download(audio_input, out=audio_file_path)
            except Exception:
                raise ValueError(f"File {audio_input} download failed.")
        else:
            logging.info("Recieved base64encoded string as input.")
            try:
                with open(audio_file_path, "wb") as audio:
                    audio.write(base64.b64decode(audio_input))
            except Exception:
                raise ValueError("Invalid audio content in base64encoded string")

        audio_nparray = audio_input_to_nparray(audio_file_path, sampling_rate)
    return audio_nparray


def predict(
    model_input: pd.DataFrame,
    task: str,
    model: WhisperForConditionalGeneration,
    tokenizer: WhisperProcessor,
    **kwargs: Dict,
) -> pd.DataFrame:
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
    :rtype: pd.DataFrame
    """
    if not task == "automatic-speech-recognition":
        return f"Invalid task name {task}"

    device = kwargs.get("device", -1)
    max_new_tokens = kwargs.get("max_new_tokens", 448)

    if device == -1 and torch.cuda.is_available():
        logging.warning('CUDA available. To switch to GPU device pass `"parameters": {"device" : 0}` in the input.')
    if device == 0 and not torch.cuda.is_available():
        device = -1
        logging.warning("CUDA unavailable. Defaulting to CPU device.")

    device = "cuda" if device == 0 else "cpu"

    logging.info(f"Using device: {device} for the inference")

    result = []
    for row in model_input.itertuples():
        # Parse inputs.

        audio = row.audio
        language = row.language or None

        if not isinstance(audio, str):
            return f"Invalid input format {type(audio)}, input should be base64 encoded string"
        if not isinstance(language, str):
            return f"Invalid language format {type(language)}, should be type string"
        if language not in supported_languages:
            return f"Language not supported. Language should be in list {supported_languages}"

        forced_decoder_ids = (
            tokenizer.get_decoder_prompt_ids(language=language, task="transcribe") if language else None
        )

        model = model.to(device)
        audio_array = audio_processor(audio)
        input_features = tokenizer(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids,
                                       max_new_tokens=max_new_tokens)
        transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        result.append({"text": transcription})

    return pd.DataFrame(result)
