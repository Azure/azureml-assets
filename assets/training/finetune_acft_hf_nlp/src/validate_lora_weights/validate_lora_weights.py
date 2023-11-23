# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate LoRA weights."""

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
import argparse
import torch
import os

COMPONENT_NAME = "ACFT-Validate_lora_weights"

logger = get_logger_app(
    "azureml.acft.contrib.hf.scripts.components.scripts.validate_lora_weights.validate_lora_weights"
)

GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "do_sample": True,
    "top_p": 0.7,
    "temperature": 0.8
}


def validate_lora_weights(base_model_path: str, config_path: str, lora_weights_path: str, tokenizer_path: str, test_examples: list):
    """Load model and make forward pass to validate lora weights."""
    logger.info("Validating lora model weights")

    config = AutoConfig.from_pretrained(config_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, config=config)

    logger.info("Loading base model")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.float16, config=config)
    logger.info("Base model loaded")

    logger.info("Loading lora adapters")
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    logger.info("lora model loaded")

    input_ids = tokenizer(test_examples, return_tensors="pt").input_ids.to("cuda")

    try:
        outputs = model.generate(inputs=input_ids, **GENERATION_CONFIG)
        predictions_text = tokenizer.batch_decode(outputs)
    except Exception:
        raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        "LoRA weights validation failed in forward pass, model cannot be inferenced in fp16"
                    )
                )
            )

    logger.info(f"inputs:\n{test_examples}\ngenerated text:\n{predictions_text}")
    logger.info("Outputs generated successfully, Validation successful")


def get_parser():
    """Get the parser object."""
    parser = argparse.ArgumentParser(description="Validate lora weights after finetuning")

    parser.add_argument(
        "--mlflow_model_path",
        type=str,
        required=True,
        help="MLFlow Model path used for validating lora weights",
    )

    parser.add_argument(
        "--lora_weights_path",
        type=str,
        required=True,
        help="LoRA weights path",
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Tokenizer path",
    )

    return parser


@swallow_all_exceptions(time_delay=5)
def main():
    """Validate lora weights after finetuning."""
    parser = get_parser()
    args, _ = parser.parse_known_args()

    set_logging_parameters(
        task_type="TextGeneration",
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
    )

    # extract model and config path from mlflow base model path
    base_model_path = os.path.join(args.mlflow_model_path, "data/model")
    config_path = os.path.join(args.mlflow_model_path, "data/config")

    validate_lora_weights(base_model_path, config_path, args.lora_weights_path, args.tokenizer_path, ["Hello"])


if __name__ == "__main__":
    main()
