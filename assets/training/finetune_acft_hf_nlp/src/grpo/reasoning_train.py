# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""GRPO trainer."""
import datasets
import json
import logging
import os
import sys
import transformers
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.utils.logging_utils import SystemSettings
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions
)
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.contrib.hf.nlp.constants.constants import (
    LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
)
from datasets import DatasetDict, load_dataset
from dataclasses import dataclass, field
from grpo_trainer_callbacks import SaveMLflowModelCallback
from grpo_trainer_rewards import get_rewards_funcs
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer, set_seed
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
)
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME

# VLLM_PP_LAYER_PARTITION = layers per pipeline stage
# VLLM_PP_NUM_PARTITIONS = number of pipeline stages (GPUs/processes)
# Both are essential for configuring pipeline parallelism in vLLM for efficient distributed training or inference.
VLLM_PP_LAYER_PARTITION = "VLLM_PP_LAYER_PARTITION"
VLLM_PP_NUM_PARTITIONS = "VLLM_PP_NUM_PARTITIONS"
os.environ[VLLM_PP_LAYER_PARTITION] = "28"
os.environ[VLLM_PP_NUM_PARTITIONS] = "8"
TASK_TYPE = 'chat-completion'
VLLM_MODE = 'colocate'  # Default vLLM mode for colocated training
COMPONENT_NAME = "ACFT-Finetune"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]
logger = get_logger_app("azureml.acft.contrib.hf.scripts.src.grpo.reasoning_train")


# System prompt used at the time of sampling
system_prompt = "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses.\
You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer.\
The reasoning process MUST BE enclosed within <think> and </think> tags."

# Chat template used for the tokenizer
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n\
{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n\
{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n\
{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n\
{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
TRAIN = 'train'
VALIDATION = 'validation'


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Extra script arguments for the GRPO training script."""

    final_model_save_path: str = field(
        default="final_model", metadata={"help": "Path to save the final model."}
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "accuracy"],
        metadata={
            "help": "List of reward functions. Possible values: 'format', 'accuracy'."
        },
    )
    mlflow_task_type: str = field(default="chat-completion")
    base_model_name: str = field(
        default="base_model", metadata={"help": "Base model name for MLflow."}
    )
    dataset_validation_split: str = field(
        default="", metadata={"help": "Path to the validation dataset."}
    )
    dataset_prompt_column: str = field(
        default="problem",
        metadata={"help": "Name of the column containing the user prompt."}
    )


@dataclass
class ExtendedGRPOConfig(GRPOConfig):
    """Extend the base GRPOConfig to add eval_strategy options."""
    eval_strategy: str = field(
        default="no",
        metadata={
            "help": "Evaluation strategy. Options: 'no', 'disable', 'steps', 'epoch'."
        }
    )

    def __post_init__(self):
        # Convert 'disable' option to 'no'
        # Need this since ev2 release pipeline doesn't allow 'no' as option
        if self.eval_strategy == "disable":
            self.eval_strategy = "no"
        # Ensure base class initialization runs (sets up distributed_state, etc.)
        try:
            super().__post_init__()  # type: ignore[attr-defined]
        except AttributeError:
            pass



def get_tokenizer(model_args: ModelConfig) -> PreTrainedTokenizer:
    """Return the tokenizer for the model.

    Args:
        model_args (ModelConfig): Model configuration object.
    Returns:
        PreTrainedTokenizer: The tokenizer for the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    return tokenizer


def make_conversation(example, dataset_prompt_column, system_prompt=None):
    """Transform the given record to be compatible for GRPO training.

    Args:
        example (dict): The input record.
        system_prompt (str): The system prompt to be used.
    Returns:
        dict: The transformed record.
    """
    prompt = []

    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})

    prompt.append({"role": "user", "content": example[dataset_prompt_column]})
    return {"prompt": prompt}


def _log_user_error(message: str):
    """Log a user error message.

    Args:
        message (str): The error message to log.
    """
    raise ACFTValidationException._with_error(
            AzureMLError.create(
                ACFTUserError,
                pii_safe_message=(
                    message
                )
            )
        )


def prepare_dataset(data_splits, dataset_prompt_column="problem", system_prompt=None):
    """Load the splits from the given dataset folder and transform the dataset to be compatible for GRPO training.

    Args:
        data_splits (list): The list containing the path of data splits.
                            Needs for train.jsonl, validation.jsonl.
        system_prompt (str): The system prompt to be used.
    Returns:
        DatasetDict: The transformed dataset.
    """
    dataset_dict = {}
    for name, split in data_splits.items():
        if os.path.exists(split):
            dataset_dict[name] = load_dataset(
                "json",
                data_files=str(split),
                split="train",
            )

    dataset = DatasetDict(dataset_dict)
    dataset = dataset.map(
        lambda example: make_conversation(example, dataset_prompt_column=dataset_prompt_column,
                                          system_prompt=system_prompt)
    )
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    return dataset


def prepare_dataset_hf(dataset: DatasetDict,
                       dataset_prompt_column: str,
                       system_prompt: str | None = None) -> DatasetDict:
    """
    Transform a pre-loaded HF DatasetDict for GRPO training.

    Args:
        dataset (DatasetDict): Hugging Face DatasetDict with train/validation/ splits.
        dataset_prompt_column (str): Name of the column containing the user prompt.
        system_prompt (str, optional): Optional system prompt to prepend.

    Returns:
        DatasetDict: The transformed dataset ready for GRPOTrainer.
    """
    def _map_fn(example):
        return make_conversation(example, dataset_prompt_column, system_prompt)

    # Apply make_conversation to each split
    prepared = DatasetDict({
        split: ds.map(_map_fn)
        for split, ds in dataset.items()
    })

    # Remove any leftover "messages" column if present
    for split in prepared:
        if "messages" in prepared[split].column_names:
            prepared[split] = prepared[split].remove_columns("messages")

    return prepared


def get_hf_model_config_and_attributes(model_path):
    """Given a HuggingFace model path, load the config and return its attributes."""
    config = AutoConfig.from_pretrained(model_path)
    return config


def main(script_args, training_args, model_args):
    """Run the GRPO training script.

    Args:
        script_args (GRPOScriptArguments): Arguments to configure the datasets and reward functions.
        training_args (GRPOConfig): Trainer-specific settings such as vLLM server config, learning rate,
        and reward weights.
        model_args (ModelConfig): Arguments to load the model.
    Returns:
        None
    """
    # Set seed for reproducibility
    set_seed(training_args.seed)

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if training_args.deepspeed:
        try:
            ds_config_path = os.path.abspath(training_args.deepspeed)
            with open(ds_config_path, "r") as fp:
                ds_config = json.load(fp)
            logger.info(f"DeepSpeed config file ({ds_config_path}) contents:\n{json.dumps(ds_config, indent=2)}")
        except Exception as e:
            logger.warning(f"Unable to read DeepSpeed config at {training_args.deepspeed}: {e}")
    else:
        logger.info("No DeepSpeed config provided in training_args.deepspeed")

    training_args.vllm_mode = VLLM_MODE

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    dataset_splits = {TRAIN: script_args.dataset_train_split, VALIDATION: script_args.dataset_validation_split}
    # If provided model path is a directory, find the subfolder containing HF model artifacts.
    root = model_args.model_name_or_path
    if os.path.isdir(root):
        for entry in os.listdir(root):
            subdir = os.path.join(root, entry)
            if not os.path.isdir(subdir):
                continue
            config_file = os.path.join(subdir, "config.json")
            tokenizer_file = os.path.join(subdir, "tokenizer.json")
            safetensors = [
                fname for fname in os.listdir(subdir)
                if fname.startswith("model-") and fname.endswith(".safetensors")
            ]
            if os.path.exists(config_file) and os.path.exists(tokenizer_file) and safetensors:
                model_args.model_name_or_path = subdir
                logger.info(f"Switched model path to subdirectory with artifacts: {subdir}")
                break
    current_policy = model_args.model_name_or_path

    # Auto Pipeline Parallel setting :
    config = get_hf_model_config_and_attributes(current_policy)
    os.environ[VLLM_PP_LAYER_PARTITION] = str(config.num_attention_heads)
    os.environ[VLLM_PP_NUM_PARTITIONS] = str(config.num_key_value_heads)

    # Load tokenizer
    tokenizer = get_tokenizer(model_args)
    # Id dataset_splits are not provided, use script_args.dataset_name and pull the dataset from the hub.
    # If only one of the splits is provided, then log a user error
    # If neither splits nor dataset_name is provided, then log a user error
    # Validate whether to use file-based splits or load from the hub
    # Check existence of dataset split files and log
    split_paths = {
        TRAIN: script_args.dataset_train_split,
        VALIDATION: script_args.dataset_validation_split,
    }
    exist_splits = {}
    for split_name, path in split_paths.items():
        if path:
            if os.path.exists(path):
                logger.info(f"{split_name.capitalize()} split file found at: {path}")
                exist_splits[split_name] = True
            else:
                logger.info(f"{split_name.capitalize()} split file not found at: {path}")
                exist_splits[split_name] = False
        else:
            exist_splits[split_name] = False

    train_split = exist_splits[TRAIN]
    val_split = exist_splits[VALIDATION]
    if train_split or val_split:
        # If any split is provided, all must be provided
        if not (train_split and val_split) and training_args.eval_strategy != "no":
            _log_user_error("When specifying dataset splits, you must provide train, and validation paths.")
            sys.exit(1)
    else:
        # No splits provided: try loading from Hugging Face Hub
        if script_args.dataset_name:
            dataset = load_dataset(script_args.dataset_name)
            logger.info(f"Loaded dataset '{script_args.dataset_name}' from the Hub.")
        else:
            _log_user_error("No dataset splits or dataset_name provided. Please specify one.")
            sys.exit(1)
    dataset_prompt_column = script_args.dataset_prompt_column or "prompt"
    # Load the dataset
    if train_split:
        dataset = prepare_dataset(dataset_splits, dataset_prompt_column=dataset_prompt_column,
                                  system_prompt=system_prompt)
        logger.info("Prepared dataset from local splits.")
    else:
        # Dataset was loaded from Hugging Face Hub, skip local file‐based preparation
        dataset = prepare_dataset_hf(dataset, dataset_prompt_column=dataset_prompt_column, system_prompt=system_prompt)
        logger.info("Preparing dataset loaded from HF hub")
    logger.info(dataset)

    # Load the reward functions
    reward_function = get_rewards_funcs(script_args.reward_funcs)
    for func in reward_function:
        logger.info(f"Using reward function: {func.__name__}")

    try:
        base_model_name = current_policy.split('/')[-1]
    except (AttributeError, TypeError):
        base_model_name = script_args.base_model_name
    # Add save callback
    save_mlflow_callback = SaveMLflowModelCallback(
        mlflow_model_save_path=script_args.final_model_save_path,
        mlflow_task_type=script_args.mlflow_task_type,
        base_model_name=base_model_name,
        processing_class=tokenizer,
    )

    # Disable evaluation if no validation data is available
    if training_args.eval_strategy != "no":
        if VALIDATION not in dataset or len(dataset[VALIDATION]) == 0:
            logger.warning("No validation set found or it's empty. Disabling evaluation.")
            training_args.eval_strategy = "no"
            eval_dataset = None
        else:
            eval_dataset = dataset[VALIDATION]
    else:
        logger.warning("No validation set found. Evaluations Disabled.")
        eval_dataset = None

    # Create the GRPOTrainer (It does SAMPLING, GRADING and TRAINING)
    trainer = GRPOTrainer(
        # The model to be trained, same copy of the model is used as reference policy.
        model=current_policy,
        # Rewards functions to be used by graders defined in "grpo_trainer_rewards.py".
        reward_funcs=reward_function,
        args=training_args,
        # Each prompt from the dataset is used to generate multiple samples.
        train_dataset=dataset[TRAIN],
        # Configuration for lora.
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
        eval_dataset=eval_dataset,
        callbacks=[save_mlflow_callback],
    )
    # Trigger the training loop
    trainer.train()


@swallow_all_exceptions(time_delay=60)
def _main():
    import faulthandler
    import os
    faulthandler.enable()
    parser = TrlParser((GRPOScriptArguments, ExtendedGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    SystemSettings.LOG_FILENAME = SystemSettings.LOG_FILENAME + f'.{os.environ["RANK"]}'
    set_logging_parameters(
        task_type=TASK_TYPE,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME,
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
        log_level=logging.INFO,
    )
    logger.info(f"model config methods: {dir(model_args)}")
    model_args.lora_target_modules = LORA_TARGET_MODULES
    # script_args has dataset_name
    main(script_args, training_args, model_args)


if __name__ == "__main__":
    try:
        _main()
    except ValueError as e:
        _log_user_error(f"ValueError occurred while running GRPO training script.{e}")
    except KeyError as e:
        _log_user_error(f"KeyError occurred while running GRPO training script. {e}")
