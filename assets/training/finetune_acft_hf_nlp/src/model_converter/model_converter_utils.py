# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for model converter utils."""

import shutil
from argparse import Namespace
from pathlib import Path

from azureml.acft.accelerator.utils.model_utils import print_model_summary
from azureml.acft.accelerator.utils.code_utils import update_json_file_and_overwrite
from azureml.acft.accelerator.constants import PeftLoRAConstants

from azureml.acft.contrib.hf.nlp.constants.constants import Tasks

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.models.auto.modeling_auto import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
)

from peft.auto import (
    AutoPeftModelForSequenceClassification,
    AutoPeftModelForTokenClassification,
    AutoPeftModelForSeq2SeqLM,
    AutoPeftModelForQuestionAnswering,
    AutoPeftModelForCausalLM,
)

from azureml.acft.common_components import get_logger_app


logger = get_logger_app(
    "azureml.acft.contrib.hf.scripts.src.model_converter.model_converter_utils"
)
from transformers import Llama4ForConditionalGeneration
from peft import PeftModel
import torch

ACFT_TASKS_HUGGINGFACE_MODELS_MAPPING = {
    Tasks.SINGLE_LABEL_CLASSIFICATION: AutoModelForSequenceClassification,
    Tasks.MULTI_LABEL_CLASSIFICATION: AutoModelForSequenceClassification,
    Tasks.NAMED_ENTITY_RECOGNITION: AutoModelForTokenClassification,
    Tasks.SUMMARIZATION: AutoModelForSeq2SeqLM,
    Tasks.TRANSLATION: AutoModelForSeq2SeqLM,
    Tasks.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    Tasks.TEXT_GENERATION: AutoModelForCausalLM,
    Tasks.CHAT_COMPLETION: AutoModelForCausalLM,
    Tasks.NLP_MULTICLASS: AutoModelForSequenceClassification,
    Tasks.NLP_MULTILABEL: AutoModelForSequenceClassification,
}


ACFT_TASKS_PEFT_MODELS_MAPPING = {
    Tasks.SINGLE_LABEL_CLASSIFICATION: AutoPeftModelForSequenceClassification,
    Tasks.MULTI_LABEL_CLASSIFICATION: AutoPeftModelForSequenceClassification,
    Tasks.NAMED_ENTITY_RECOGNITION: AutoPeftModelForTokenClassification,
    Tasks.SUMMARIZATION: AutoPeftModelForSeq2SeqLM,
    Tasks.TRANSLATION: AutoPeftModelForSeq2SeqLM,
    Tasks.QUESTION_ANSWERING: AutoPeftModelForQuestionAnswering,
    Tasks.TEXT_GENERATION: AutoPeftModelForCausalLM,
    Tasks.CHAT_COMPLETION: AutoPeftModelForCausalLM,
}


def load_tokenizer(
    tokenizer_path: str, component_args: Namespace, ft_config: dict
) -> PreTrainedTokenizerBase:
    """Load ACFT tokenizer."""
    load_tokenizer_kwargs = ft_config.get("load_tokenizer_kwargs", {})
    logger.info(f"Loading tokenizer with kwargs: {load_tokenizer_kwargs}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        **load_tokenizer_kwargs,
    )
    return tokenizer


def load_model(
    model_path: str, component_args: Namespace, ft_config: dict
) -> PreTrainedModel:
    """Load ACFT model."""
    is_peft_lora_model = getattr(component_args, "apply_lora", False) and getattr(
        component_args, "lora_algo", False
    )
    is_ds3_model = (
        getattr(component_args, "apply_deepspeed", False)
        and getattr(component_args, "deepspeed_stage", 2) == 3
    )

    # if the model is Deepspeed stage-3 PEFT LoRA model the layers needs to be merged
    if is_peft_lora_model and is_ds3_model:
        model = load_and_merge_peft_lora_model(model_path, component_args, ft_config)
    else:
        auto_cls = ACFT_TASKS_HUGGINGFACE_MODELS_MAPPING[component_args.task_name]
        logger.info(f"Identified auto cls: {auto_cls}")
        # load model in cpu as in gpu model weight tensors could have shared memory
        load_model_kwargs = {
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
            "torch_dtype": "auto",
        }
        load_model_kwargs.update(ft_config.get("load_model_kwargs", {}))
        # pop any optimizations related parameters
        load_model_kwargs.pop("attn_implementation", None)
        logger.info(f"Loading model with kwargs: {load_model_kwargs}")
        if "llama-4" in model_path.lower():
            model = Llama4ForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                **load_model_kwargs,
            )
        else:
            model = auto_cls.from_pretrained(
                model_path,
                **load_model_kwargs,
            )
    print_model_summary(model, True)

    return model  # type: ignore


def load_and_merge_peft_lora_model(
    model_path: str, component_args: Namespace, ft_config: dict
) -> PreTrainedModel:
    """Load and merge ACFT PEFT LoRA model layers."""
    peft_lora_adapter_path = str(
        Path(model_path, PeftLoRAConstants.PEFT_ADAPTER_WEIGHTS_FOLDER)
    )
    # copy PEFT_ADAPTER_WEIGHTS_FOLDER to ACFT_PEFT_CHECKPOINT_PATH and
    # change `base_model_path` in peft adapter_config to model_path
    if Path(PeftLoRAConstants.ACFT_PEFT_CHECKPOINT_PATH).is_dir():
        shutil.rmtree(PeftLoRAConstants.ACFT_PEFT_CHECKPOINT_PATH)
    # Path(PeftLoRAConstants.ACFT_PEFT_CHECKPOINT_PATH).mkdir(exist_ok=True, parents=True)
    shutil.copytree(
        peft_lora_adapter_path,
        PeftLoRAConstants.ACFT_PEFT_CHECKPOINT_PATH,
        dirs_exist_ok=True,
    )
    peft_lora_adapter_path = PeftLoRAConstants.ACFT_PEFT_CHECKPOINT_PATH
    peft_adapter_config_file = str(
        Path(peft_lora_adapter_path, PeftLoRAConstants.PEFT_ADAPTER_CONFIG_FILE_NAME)
    )
    update_config = {
        PeftLoRAConstants.PEFT_LORA_BASE_MODEL_PATH_KEY: model_path,
    }
    update_json_file_and_overwrite(peft_adapter_config_file, update_config)

    # peft model loading expects tokenizer in peft adapter path (peft>=0.8.0)
    tokenizer = load_tokenizer(model_path, component_args, ft_config)
    tokenizer.save_pretrained(peft_lora_adapter_path)

    # Load the PEFT model
    auto_cls = ACFT_TASKS_PEFT_MODELS_MAPPING[component_args.task_name]
    logger.info(f"Identified auto cls: {auto_cls}")
    # load model in cpu as in gpu model weight tensors could have shared memory
    load_model_kwargs = {"device_map": "cpu", "low_cpu_mem_usage": True}
    load_model_kwargs.update(ft_config.get("load_model_kwargs", {}))
    # pop any optimizations related parameters
    load_model_kwargs.pop("attn_implementation", None)
    logger.info(f"Loading model with kwargs: {load_model_kwargs}")
    if "llama-4" in model_path.lower():
        base_model = Llama4ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, trust_remote_code=True
        )
        ds3_peft_model = PeftModel.from_pretrained(
            base_model,
            peft_lora_adapter_path,
            **load_model_kwargs,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
    else:
        ds3_peft_model = auto_cls.from_pretrained(
            peft_lora_adapter_path,
            **load_model_kwargs,
        )  # loads both base weights and lora weights
    # merge LoRA weights
    merged_model = ds3_peft_model.merge_and_unload()
    merged_model = merged_model.to(torch.float16)
    logger.info("Merged LoRA layers.")
    return merged_model


def copy_tokenizer_files_to_model_folder(mlflow_model_folder: str, task_name: str):
    """Copy tokenizer files to model folder.

    It expects tokenizer and model files in same folder i.e. "data/model".
    """
    src_dir = Path(mlflow_model_folder, "data", "tokenizer")
    dst_dir = Path(mlflow_model_folder, "data", "model")
    if src_dir.is_dir() and dst_dir.is_dir():
        logger.info("Copying tokenizer files to model folder for {}".format(task_name))
        shutil.copytree(
            str(Path(mlflow_model_folder, "data", "tokenizer")),
            str(Path(mlflow_model_folder, "data", "model")),
            dirs_exist_ok=True,
        )
        logger.info(
            "Copy completed for tokenizer files to model folder for {}".format(
                task_name
            )
        )
    else:
        logger.warning("Couldn't copy the tokenizer files to model folder.")
