# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Evaluations Component.

This component:
1. Lists the directory structure of checkpoints
2. Runs evaluations through multiple checkpoints based on explore_pattern
3. Evaluates each checkpoint using vLLM with step-based logging
"""

import argparse
import json
import os
import sys
from datetime import datetime

from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError, ACFTSystemError
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS

# Import functions from other modules instead of subprocess calls
from list_checkpoint_dirs import list_directory_contents, format_size

# Import vLLM evaluation - will be imported at runtime to avoid import errors
# from vllm_evaluation_step_logging import run_evaluation

TASK_TYPE = 'llm-evaluation'
COMPONENT_NAME = "ACFT-LLM-Evaluation"

# File and configuration constants
CONFIG_FILENAME = "config.json"
MODEL_SAFETENSORS_INDEX = "model.safetensors.index.json"
AGGREGATE_METRICS_FILENAME = "aggregate_metrics.json"

# Mode strings
MODE_FULL_CHECKPOINT = "Mode: Full Checkpoint"
MODE_LORA_ADAPTER = "Mode: LoRA Adapter"
MODE_BASE_MODEL = "Mode: Base Model Evaluation"

# Checkpoint field names
FIELD_CHECKPOINT_VALUE = "checkpoint_value"
FIELD_SOURCE_LABEL = "source_label"
FIELD_BASE_MODEL = "base_model"
FIELD_USE_LORA = "use_lora"

logger = get_logger_app("azureml.acft.contrib.hf.scripts.src.llm_evaluation.model_evaluation")


def str2bool(v):
    """Convert string to boolean for argparse compatibility."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _log_user_error(message: str):
    """Log a user error and raise ACFTValidationException.

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


def _log_system_error(message: str):
    """Log a system error and raise ACFTValidationException.

    Args:
        message (str): The error message to log.
    """
    raise ACFTValidationException._with_error(
        AzureMLError.create(
            ACFTSystemError,
            pii_safe_message=(
                message
            )
        )
    )


def log_section_header(title: str):
    """Log section header with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"\n{'='*80}")
    logger.info(f"[{timestamp}] {title}")
    logger.info(f"{'='*80}\n")


def get_azureml_run():
    """Get AzureML Run context if available."""
    try:
        from azureml.core.run import Run
        azureml_run = Run.get_context()
        if azureml_run and "OfflineRun" not in azureml_run.id:
            return azureml_run
    except ImportError:
        logger.info("Warning: azureml-core not available - AzureML logging disabled")
    except Exception as e:
        logger.info(f"Warning: Failed to get AzureML run context: {e}")
    return None


def list_checkpoint_directory_structure(base_path: str, max_depth: int = 3):
    """List and print the checkpoint directory structure."""
    log_section_header("Listing Checkpoint Directory Structure")

    logger.info(f"Base Path: {base_path}")
    logger.info(f"Max Depth: {max_depth}")
    try:
        if not os.path.exists(base_path):
            _log_user_error(f"Path does not exist: {base_path}")

        logger.info(f"Path exists: {base_path}")
        logger.info(f"Is Directory: {os.path.isdir(base_path)}")

        if os.path.isdir(base_path):
            logger.info("\nDirectory Contents:\n")
            list_directory_contents(base_path, max_depth=max_depth)

            # Count files and directories
            total_files = 0
            total_dirs = 0
            total_size = 0

            for root, dirs, files in os.walk(base_path):
                total_dirs += len(dirs)
                total_files += len(files)
                for file in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, file))
                    except Exception:
                        # Ignore files that can't be accessed
                        pass

            logger.info(f"\n{'='*80}")
            logger.info("Summary:")
            logger.info(f"Total Directories: {total_dirs}")
            logger.info(f"Total Files: {total_files}")
            logger.info(f"Total Size: {format_size(total_size)}")
        else:
            size = os.path.getsize(base_path)
            logger.info(f"\nFile Size: {format_size(size)}")
    except Exception as e:
        error_msg = f"Error listing directory: {e}"
        _log_system_error(error_msg)


def find_model_directory(base_path: str, max_depth: int = 3) -> str:
    """
    Recursively search for a directory containing config.json.

    Returns the path if found, otherwise returns the original base_path.
    """
    if not os.path.exists(base_path):
        return base_path

    # Check if current directory has config.json
    config_path = os.path.join(base_path, CONFIG_FILENAME)
    if os.path.exists(config_path):
        return base_path

    # Recursively search subdirectories
    def search_subdirs(current_path: str, depth: int = 0) -> str:
        if depth >= max_depth:
            return None

        try:
            for entry in os.listdir(current_path):
                entry_path = os.path.join(current_path, entry)
                if os.path.isdir(entry_path):
                    config_path = os.path.join(entry_path, CONFIG_FILENAME)
                    if os.path.exists(config_path):
                        return entry_path

                    # Recurse into subdirectory
                    result = search_subdirs(entry_path, depth + 1)
                    if result:
                        return result
        except (PermissionError, OSError):
            # Ignoring directories that can't be accessed
            pass

        return None

    result = search_subdirs(base_path)
    return result if result else base_path


def resolve_checkpoint_path(base_path: str, pattern: str, checkpoint_value: str) -> str:
    """Resolve the full checkpoint path from pattern and value."""
    # Replace {checkpoint} placeholder with actual value
    relative_path = pattern.replace("{checkpoint}", str(checkpoint_value))
    full_path = os.path.join(base_path, relative_path)
    return full_path


def evaluate_checkpoint(
    checkpoint_path: str,
    checkpoint_value: str,
    validation_file: str,
    output_dir: str,
    intermediate_dir: str,
    use_lora_adapters: bool,
    base_model_path: str,
    hf_model_id: str,
    args: argparse.Namespace,
    azureml_run,
    checkpoint_source: str = "base_path_1",
    checkpoint_source_label: str = "base_path_1"
):
    """Evaluate a single checkpoint using vLLM."""
    logger.info(f"Evaluating Checkpoint: {checkpoint_value}")
    logger.info(f"Source: {checkpoint_source}")

    # Handle base model evaluation case
    if checkpoint_source == FIELD_BASE_MODEL:
        logger.info(MODE_BASE_MODEL)
        if hf_model_id:
            logger.info(f"Base Model (HF): {hf_model_id}")
        elif base_model_path:
            logger.info(f"Base Model (Local): {base_model_path}")
    elif use_lora_adapters:
        logger.info(MODE_LORA_ADAPTER)
        if hf_model_id:
            logger.info(f"Base Model (HF): {hf_model_id}")
        elif base_model_path:
            logger.info(f"Base Model (Local): {base_model_path}")
        logger.info(f"LoRA Adapter: {checkpoint_path}")
    else:
        logger.info(MODE_FULL_CHECKPOINT)
        logger.info(f"Checkpoint path: {checkpoint_path if checkpoint_path else 'Using HF Model ID'}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Intermediate directory: {intermediate_dir}")

    # Handle base model evaluation
    if checkpoint_source == FIELD_BASE_MODEL:
        # Base model is ALWAYS evaluated as full weights, not LoRA
        logger.info(r"Base model evaluation always uses full weights inference (ignoring use_lora_adapters flag)")

        # Determine which base model to use
        if base_model_path and hf_model_id:
            logger.warning(r"Both base_model_path and hf_model_id provided. Using base_model_path")
            actual_base_model = base_model_path
        elif hf_model_id:
            logger.info(f"Using Hugging Face model: {hf_model_id}")
            actual_base_model = hf_model_id
        elif base_model_path:
            # Recursively search for config.json in base_model_path
            logger.info(f"Searching for model directory with config.json in: {base_model_path}")
            resolved_base_model = find_model_directory(base_model_path)

            if resolved_base_model != base_model_path:
                logger.info(f"Found model directory: {resolved_base_model}")
                actual_base_model = resolved_base_model
            else:
                actual_base_model = base_model_path

            if not os.path.exists(actual_base_model):
                logger.error(f"Base model path does not exist: {actual_base_model}")
                return False

            config_path = os.path.join(actual_base_model, CONFIG_FILENAME)
            if not os.path.exists(config_path):
                logger.warning(f"config.json not found at: {config_path}")
            else:
                logger.info(f"Base model found with config.json: {actual_base_model}")
        else:
            logger.error(r"Either base_model_path or hf_model_id is required for base model evaluation")
            return False

        # For base model: always use full weights (no LoRA)
        eval_checkpoint_path = actual_base_model
        eval_base_model = None
        use_lora_for_eval = False
    # Validate paths/IDs based on mode
    elif use_lora_adapters:
        # LoRA mode: checkpoint_path must exist, and base model is required
        if not os.path.exists(checkpoint_path):
            logger.error(f"Adapter path does not exist: {checkpoint_path}")
            logger.info(f"Skipping checkpoint {checkpoint_value}")
            return False
        # LoRA mode validation
        # Validate that either base_model_path or hf_model_id is provided
        if not base_model_path and not hf_model_id:
            logger.error(r"Either base_model_path or hf_model_id is required when use_lora_adapters is true")
            return False

        # Validate mutual exclusivity
        if base_model_path and hf_model_id:
            logger.warning(r"Both base_model_path and hf_model_id provided.\
                Using base_model_path and ignoring hf_model_id")
            # Recursively search for config.json in base_model_path
            logger.info(f"Searching for model directory with config.json in: {base_model_path}")
            resolved_base_model = find_model_directory(base_model_path)

            if resolved_base_model != base_model_path:
                logger.info(f"Found model directory: {resolved_base_model}")

            actual_base_model = resolved_base_model
        elif hf_model_id:
            logger.info(f"Using Hugging Face model: {hf_model_id}")
            actual_base_model = hf_model_id
        else:
            # Recursively search for config.json in base_model_path
            logger.info(f"Searching for model directory with config.json in: {base_model_path}")
            resolved_base_model = find_model_directory(base_model_path)

            if resolved_base_model != base_model_path:
                logger.info(f"Found model directory: {resolved_base_model}")
                actual_base_model = resolved_base_model
            else:
                actual_base_model = base_model_path

            # Validate local path exists
            if not os.path.exists(actual_base_model):
                logger.error(f"Base model path does not exist: {actual_base_model}")
                return False

            config_path = os.path.join(actual_base_model, CONFIG_FILENAME)
            if not os.path.exists(config_path):
                logger.warning(f"config.json not found at: {config_path}")
            else:
                logger.info(f"Base model found with config.json: {actual_base_model}")

        logger.info(f"LoRA adapter found: {checkpoint_path}")

        # For LoRA: use adapter path and base model
        eval_checkpoint_path = checkpoint_path
        eval_base_model = actual_base_model
        use_lora_for_eval = True
    else:
        # Full model mode: use checkpoint_path if exists, otherwise use HF model
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Local checkpoint exists - use it directly
            log_section_header(f"Evaluating Checkpoint {checkpoint_value}")

            # Path to use for evaluation
            eval_checkpoint_path = checkpoint_path
            eval_base_model = None
            use_lora_for_eval = False
        elif hf_model_id:
            # No local checkpoint - use HF model directly
            logger.info(f"Using Hugging Face model directly: {hf_model_id}")
            eval_checkpoint_path = hf_model_id
            eval_base_model = None
            use_lora_for_eval = False
        else:
            logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
            return False

    # Create checkpoint-specific output directory with source identifier
    checkpoint_output_dir = os.path.join(output_dir, f"{checkpoint_source}_checkpoint_{checkpoint_value}")
    os.makedirs(checkpoint_output_dir, exist_ok=True)

    # Run vLLM evaluation directly
    log_section_header(f"EVALUATING CHECKPOINT {checkpoint_value}")
    logger.info("Evaluation Details:")
    if checkpoint_source == FIELD_BASE_MODEL:
        logger.info("Mode: Base Model (Full Weights)")
        logger.info(f"Model Path: {eval_checkpoint_path}")
    elif use_lora_for_eval:
        logger.info(MODE_LORA_ADAPTER)
        logger.info(f"Base Model: {eval_base_model}")
        logger.info(f"LoRA Adapter: {eval_checkpoint_path}")
    else:
        logger.info(MODE_FULL_CHECKPOINT)
        logger.info(f"Original Path: {checkpoint_path}")
        logger.info(f"Eval Path: {eval_checkpoint_path}")
    logger.info(f"Output Dir: {checkpoint_output_dir}")
    logger.info(f"Trials: {args.number_of_trials}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Temperature: {args.temperature}")

    try:
        # Import and run vLLM evaluation directly
        sys.path.insert(0, os.path.dirname(__file__))
        from vllm_evaluation_step_logging import main as vllm_main

        # Save original sys.argv
        original_argv = sys.argv

        # Create arguments for vLLM evaluation
        sys_argv_list = [
            'vllm_evaluation_step_logging.py',
            '--model_path', eval_checkpoint_path,
            '--validation_file', validation_file,
            '--max_prompt_length', str(args.max_prompt_length),
            '--max_response_length', str(args.max_response_length),
            '--batch_size', str(args.batch_size),
            '--temperature', str(args.temperature),
            '--top_p', str(args.top_p),
            '--tensor_parallel_size', str(args.tensor_parallel_size),
            '--gpu_memory_utilization', str(args.gpu_memory_utilization),
            '--dtype', args.dtype,
            '--extraction_method', args.extraction_method,
            '--n_gpus_per_node', str(args.n_gpus_per_node),
            '--number_of_trials', str(args.number_of_trials),
            '--output_dir', checkpoint_output_dir
        ]

        # Add LoRA-specific arguments if using adapters
        if use_lora_for_eval:
            sys_argv_list.extend([
                '--enable_lora',
                '--lora_modules', f'adapter={eval_checkpoint_path}',
                '--max_lora_rank', str(args.max_lora_rank)
            ])
            # Override model_path to be the base model
            sys_argv_list[sys_argv_list.index('--model_path') + 1] = eval_base_model

        sys.argv = sys_argv_list

        # Call vLLM evaluation main function
        vllm_main()

        # Restore original sys.argv
        sys.argv = original_argv

        logger.info(f"Successfully evaluated checkpoint {checkpoint_value}")

        # Load and log checkpoint-level aggregate metrics to AzureML
        if azureml_run:
            try:
                aggregate_metrics_path = os.path.join(checkpoint_output_dir, AGGREGATE_METRICS_FILENAME)
                if os.path.exists(aggregate_metrics_path):
                    with open(aggregate_metrics_path, 'r') as f:
                        agg_metrics = json.load(f)

                    # Log aggregate metrics with checkpoint identifier and custom source label
                    azureml_run.log(f"{checkpoint_source_label}/checkpoint_{checkpoint_value}/accuracy_mean",
                                    agg_metrics['accuracy']['mean'])
                    azureml_run.log(f"{checkpoint_source_label}/checkpoint_{checkpoint_value}/format_rate_mean",
                                    agg_metrics['format_rate']['mean'])
                    azureml_run.log(f"{checkpoint_source_label}/checkpoint_{checkpoint_value}/correct_answers_mean",
                                    agg_metrics['correct_answers']['mean'])

                    if agg_metrics['number_of_trials'] > 1:
                        azureml_run.log(f"{checkpoint_source_label}/checkpoint_{checkpoint_value}/accuracy_std",
                                        agg_metrics['accuracy']['std'])

                        # Calculate and log min and max accuracy values across trials
                        if 'values' in agg_metrics['accuracy'] and len(agg_metrics['accuracy']['values']) > 0:
                            azureml_run.log(f"{checkpoint_source_label}/checkpoint_{checkpoint_value}/min",
                                            min(agg_metrics['accuracy']['values']))
                            azureml_run.log(f"{checkpoint_source_label}/checkpoint_{checkpoint_value}/max",
                                            max(agg_metrics['accuracy']['values']))

                    logger.info(f"Logged checkpoint {checkpoint_value} (source: {checkpoint_source},\
                                label: {checkpoint_source_label}) aggregate metrics to AzureML")
            except Exception as e:
                logger.info(f"Warning: Failed to log checkpoint metrics to AzureML: {e}")

        return True

    except Exception as e:
        logger.error(f"Failed to evaluate checkpoint {checkpoint_value}")
        logger.info(f"Error: {e}")
        return False


def main():
    """Execute the model evaluation component."""
    parser = argparse.ArgumentParser(description="Evaluation Checkpoint Evaluations Component")

    # Checkpoint eval parameters
    parser.add_argument("--checkpoint_base_path_1", type=str, default=None,
                        help="Base path containing all checkpoints or LoRA adapters\
                            (optional if using hf_model_id for full models)")
    parser.add_argument("--checkpoint_base_path_2", type=str, default=None,
                        help="Second base path containing checkpoints or LoRA adapters\
                            (optional, for comparing models from different training runs)")
    parser.add_argument("--base_path_1_label", type=str, default="base_path_1",
                        help="Label to use as prefix in metrics for checkpoint_base_path_1\
                            (e.g., 'experiment_a'). Defaults to 'base_path_1'")
    parser.add_argument("--base_path_2_label", type=str, default="base_path_2",
                        help="Label to use as prefix in metrics for checkpoint_base_path_2\
                            (e.g., 'experiment_b'). Defaults to 'base_path_2'")
    parser.add_argument("--evaluate_base_model", type=str2bool, nargs='?', const=True, default=False,
                        help="If true, also evaluate the base model after evaluating checkpoints\
                            (accepts: True/False or flag)")
    parser.add_argument("--explore_pattern_1", type=str,
                        default="global_step_{checkpoint}/actor/huggingface/",
                        help="Pattern to explore for checkpoint paths (only used with checkpoint_base_path_1)")
    parser.add_argument("--explore_pattern_2", type=str,
                        default="global_step_{checkpoint}/actor/huggingface/",
                        help="Pattern to explore for checkpoint paths in checkpoint_base_path_2\
                            (only used with checkpoint_base_path_2)")
    parser.add_argument("--checkpoint_values_1", type=str, default=None,
                        help="Comma-separated list of checkpoint values (e.g., '100,129,20').\
                            Optional if using only hf_model_id")
    parser.add_argument("--checkpoint_values_2", type=str, default=None,
                        help="Comma-separated list of checkpoint values for \
                            checkpoint_base_path_2 (e.g., '100,129,20'). Only used with checkpoint_base_path_2")

    # LoRA-specific parameters
    parser.add_argument("--use_lora_adapters_1", type=str2bool, nargs='?', const=True, default=False,
                        help="If true, checkpoints from checkpoint_base_path_1 are LoRA adapters\
                            to load with base model (accepts: True/False or flag)")
    parser.add_argument("--use_lora_adapters_2", type=str2bool, nargs='?', const=True, default=False,
                        help="If true, checkpoints from checkpoint_base_path_2 are LoRA adapters\
                            to load with base model (accepts: True/False or flag)")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Local base model path (mutually exclusive with hf_model_id)")
    parser.add_argument("--hf_model_id", type=str, default=None,
                        help="Hugging Face model ID (e.g., 'microsoft/Phi-4-reasoning',\
                            mutually exclusive with base_model_path)")
    parser.add_argument("--max_lora_rank", type=int, default=64,
                        help="Maximum LoRA rank for adapter support (default: 64)")

    # Evaluation parameters
    parser.add_argument("--validation_file", type=str, required=True,
                        help="Path to validation JSONL file")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_response_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--extraction_method", type=str, default="strict",
                        choices=["strict", "flexible"])
    parser.add_argument("--n_gpus_per_node", type=int, default=1)
    parser.add_argument("--number_of_trials", type=int, default=1)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for all evaluation results")
    parser.add_argument("--intermediate_dir", type=str, required=True,
                        help="Intermediate directory for preprocessed checkpoints")

    args = parser.parse_args()

    # Set logging parameters
    set_logging_parameters(
        task_type=TASK_TYPE,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
    )

    # Validate configuration
    if args.use_lora_adapters_1:
        # LoRA mode: requires checkpoint_base_path_1 and a base model
        if not args.checkpoint_base_path_1:
            _log_user_error("--checkpoint_base_path_1 is required when --use_lora_adapters_1 is true")
        if not args.checkpoint_values_1:
            _log_user_error("--checkpoint_values_1 is required when --use_lora_adapters_1 is true")
        if not args.base_model_path and not args.hf_model_id:
            _log_user_error("Either --base_model_path or --hf_model_id is required when --use_lora_adapters_1 is true")
        if args.base_model_path and args.hf_model_id:
            logger.warning("Both --base_model_path and --hf_model_id provided. Using --base_model_path")

    if args.use_lora_adapters_2:
        # LoRA mode: requires checkpoint_base_path_2 and a base model
        if not args.checkpoint_base_path_2:
            _log_user_error("--checkpoint_base_path_2 is required when --use_lora_adapters_2 is true")
        if not args.checkpoint_values_2:
            _log_user_error("--checkpoint_values_2 is required when --use_lora_adapters_2 is true")
        if not args.base_model_path and not args.hf_model_id:
            _log_user_error("Either --base_model_path or --hf_model_id is required when --use_lora_adapters_2 is true")
        if args.base_model_path and args.hf_model_id:
            logger.warning("Both --base_model_path and --hf_model_id provided. Using --base_model_path")

    if not args.use_lora_adapters_1 and not args.use_lora_adapters_2:
        # Full model mode: need either checkpoints OR hf_model_id
        if not args.checkpoint_base_path_1 and not args.hf_model_id:
            _log_user_error("Either --checkpoint_base_path_1 or --hf_model_id is required for full model evaluation")

        # If using checkpoints, need checkpoint_values_1
        if args.checkpoint_base_path_1 and not args.checkpoint_values_1:
            _log_user_error("--checkpoint_values_1 is required when using checkpoint_base_path_1")

        # If using HF model only, checkpoint_values_1 should default to a single dummy value
        if not args.checkpoint_base_path_1 and args.hf_model_id:
            if not args.checkpoint_values_1:
                args.checkpoint_values_1 = "hf_model"
                logger.info("Using HuggingFace model directly, setting checkpoint_values_1 to 'hf_model'")

        if args.checkpoint_base_path_1 and args.hf_model_id:
            logger.info("Both --checkpoint_base_path_1 and --hf_model_id provided. Using --checkpoint_base_path_1")

    # Validate evaluate_base_model option
    if args.evaluate_base_model:
        if not args.base_model_path and not args.hf_model_id:
            _log_user_error("Either --base_model_path or --hf_model_id is required when --evaluate_base_model is true")

    # Initialize AzureML Run context
    azureml_run = get_azureml_run()
    if azureml_run:
        logger.info("AzureML Run context found - checkpoint metrics will be logged")

    # Print the actual values
    # Formatting with = only for visual separation in logs to user, saving space in ms telemetry by not using logger
    print("=" * 80)
    logger.info(f"use_lora_adapters_1 = {args.use_lora_adapters_1} (type: {type(args.use_lora_adapters_1).__name__})")
    logger.info(f"use_lora_adapters_2 = {args.use_lora_adapters_2} (type: {type(args.use_lora_adapters_2).__name__})")
    logger.info(f"checkpoint_base_path_1 = {args.checkpoint_base_path_1}")
    logger.info(f"checkpoint_base_path_2 = {args.checkpoint_base_path_2}")
    logger.info(f"hf_model_id = {args.hf_model_id}")
    logger.info(f"base_model_path = {args.base_model_path}")
    logger.info(f"evaluate_base_model = {args.evaluate_base_model}")

    logger.info(f"Mode base_path_1: {'LoRA Adapters' if args.use_lora_adapters_1 else 'Full Checkpoints'}")
    logger.info(f"Mode base_path_2: {'LoRA Adapters' if args.use_lora_adapters_2 else 'Full Checkpoints'}")
    if args.use_lora_adapters_1 or args.use_lora_adapters_2:
        if args.hf_model_id:
            logger.info(f"Base model (HuggingFace): {args.hf_model_id}")
        if args.base_model_path:
            logger.info(f"Base model (Local): {args.base_model_path}")
        if args.checkpoint_base_path_1:
            logger.info(f"Adapter base path 1: {args.checkpoint_base_path_1}")
        if args.checkpoint_base_path_2:
            logger.info(f"Adapter base path 2: {args.checkpoint_base_path_2}")
    else:
        if args.checkpoint_base_path_1:
            logger.info(f"Checkpoint base path 1: {args.checkpoint_base_path_1}")
        if args.checkpoint_base_path_2:
            logger.info(f"Checkpoint base path 2: {args.checkpoint_base_path_2}")
        if args.hf_model_id:
            logger.info(f"Using HuggingFace model: {args.hf_model_id}")
    if args.checkpoint_base_path_1:
        logger.info(f"Explore pattern 1: {args.explore_pattern_1}")
        logger.info(f"Checkpoint values 1: {args.checkpoint_values_1}")
    if args.checkpoint_base_path_2:
        logger.info(f"Explore pattern 2: {args.explore_pattern_2}")
        logger.info(f"Checkpoint values 2: {args.checkpoint_values_2}")
    if args.evaluate_base_model:
        logger.info(r"Evaluate base model: Yes")
    logger.info(f"Validation file: {args.validation_file}")
    logger.info(f"Number of trials per checkpoint: {args.number_of_trials}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Intermediate directory: {args.intermediate_dir}")
    print("=" * 80)

    # Create output and intermediate directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.intermediate_dir, exist_ok=True)

    # Step 1: List directory structures
    if args.checkpoint_base_path_1:
        logger.info("\n[STEP 1a] Listing checkpoint directory structure (base_path_1)...")
        list_checkpoint_directory_structure(args.checkpoint_base_path_1, max_depth=3)

    if args.checkpoint_base_path_2:
        logger.info("\n[STEP 1b] Listing checkpoint directory structure (base_path_2)...")
        list_checkpoint_directory_structure(args.checkpoint_base_path_2, max_depth=3)

    if not args.checkpoint_base_path_1 and not args.checkpoint_base_path_2:
        logger.info("\n[STEP 1] Skipping directory listing (using HuggingFace model)")

    # Step 2: Build evaluation tasks list
    logger.info("\n[STEP 2] Building evaluation tasks...")
    eval_tasks = []

    # Add base model evaluation task FIRST if requested
    if args.evaluate_base_model:
        logger.info("Adding base model evaluation task (will run first)")
        eval_tasks.append({
            FIELD_CHECKPOINT_VALUE: FIELD_BASE_MODEL,
            "base_path": None,
            "pattern": None,
            "source": FIELD_BASE_MODEL
        })

    # Add tasks from checkpoint_base_path_1
    if args.checkpoint_values_1:
        checkpoint_values_1 = [val.strip() for val in args.checkpoint_values_1.split(',')]
        logger.info(f"Found {len(checkpoint_values_1)} checkpoints from base_path_1: {checkpoint_values_1}")
        for checkpoint_value in checkpoint_values_1:
            eval_tasks.append({
                FIELD_CHECKPOINT_VALUE: checkpoint_value,
                "base_path": args.checkpoint_base_path_1,
                "pattern": args.explore_pattern_1,
                "source": "base_path_1",
                FIELD_SOURCE_LABEL: args.base_path_1_label,
                FIELD_USE_LORA: args.use_lora_adapters_1
            })

    # Add tasks from checkpoint_base_path_2
    if args.checkpoint_values_2:
        checkpoint_values_2 = [val.strip() for val in args.checkpoint_values_2.split(',')]
        logger.info(f"Found {len(checkpoint_values_2)} checkpoints from base_path_2: {checkpoint_values_2}")
        for checkpoint_value in checkpoint_values_2:
            eval_tasks.append({
                FIELD_CHECKPOINT_VALUE: checkpoint_value,
                "base_path": args.checkpoint_base_path_2,
                "pattern": args.explore_pattern_2,
                "source": "base_path_2",
                FIELD_SOURCE_LABEL: args.base_path_2_label,
                FIELD_USE_LORA: args.use_lora_adapters_2
            })

    logger.info(f"Total evaluation tasks: {len(eval_tasks)}")

    # Step 3: eval through checkpoints and evaluate
    logger.info(f"\n[STEP 3] Evaluating through {len(eval_tasks)} checkpoint(s)...")

    results_summary = []
    successful_evals = 0
    failed_evals = 0

    for idx, task in enumerate(eval_tasks, 1):
        checkpoint_value = task[FIELD_CHECKPOINT_VALUE]
        base_path = task["base_path"]
        pattern = task["pattern"]
        source = task["source"]
        source_label = task.get(FIELD_SOURCE_LABEL, source)  # Fallback to source if no label provided
        use_lora = task.get(FIELD_USE_LORA, False)

        logger.info(f"Processing checkpoint {idx}/{len(eval_tasks)}: {checkpoint_value} \
            (source: {source}, label: {source_label})")

        # Resolve checkpoint path (None if using HF model directly)
        if base_path:
            checkpoint_path = resolve_checkpoint_path(
                base_path,
                pattern,
                checkpoint_value
            )
        else:
            checkpoint_path = None

        # Evaluate checkpoint
        success = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoint_value=checkpoint_value,
            validation_file=args.validation_file,
            output_dir=args.output_dir,
            intermediate_dir=args.intermediate_dir,
            use_lora_adapters=use_lora,
            base_model_path=args.base_model_path,
            hf_model_id=args.hf_model_id,
            args=args,
            azureml_run=azureml_run,
            checkpoint_source=source,
            checkpoint_source_label=source_label
        )

        if success:
            successful_evals += 1
        else:
            failed_evals += 1

        results_summary.append({
            FIELD_CHECKPOINT_VALUE: checkpoint_value,
            "checkpoint_path": checkpoint_path,
            "source": source,
            "success": success,
            "output_dir": os.path.join(args.output_dir, f"{source}_checkpoint_{checkpoint_value}")
        })

    # Step 4: Save evaluation summary
    print("\n" + "=" * 80)
    logger.info("Evaluation Summary:")
    print("=" * 80)
    logger.info(f"Total checkpoints: {len(eval_tasks)}")
    logger.info(f"Successful evaluations: {successful_evals}")
    logger.info(f"Failed evaluations: {failed_evals}")
    print("=" * 80)

    # Log summary to AzureML
    if azureml_run:
        try:
            azureml_run.log("eval/total_checkpoints", len(eval_tasks))
            azureml_run.log("eval/successful_evaluations", successful_evals)
            azureml_run.log("eval/failed_evaluations", failed_evals)
            logger.info("Logged eval summary to AzureML")
        except Exception as e:
            logger.info(f"Warning: Failed to log eval summary to AzureML: {e}")

    # Save results summary
    summary_data = {
        "config": {
            "checkpoint_base_path_1": args.checkpoint_base_path_1,
            "checkpoint_base_path_2": args.checkpoint_base_path_2,
            "explore_pattern_1": args.explore_pattern_1,
            "explore_pattern_2": args.explore_pattern_2,
            "checkpoint_values_1": args.checkpoint_values_1,
            "checkpoint_values_2": args.checkpoint_values_2,
            "evaluate_base_model": args.evaluate_base_model,
            "use_lora_adapters_1": args.use_lora_adapters_1,
            "use_lora_adapters_2": args.use_lora_adapters_2,
            "validation_file": args.validation_file,
            "number_of_trials": args.number_of_trials
        },
        "summary": {
            "total_checkpoints": len(eval_tasks),
            "successful_evaluations": successful_evals,
            "failed_evaluations": failed_evals
        },
        "results": results_summary
    }

    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"\n Evaluation summary saved to: {summary_path}")

    # Print individual checkpoint results
    logger.info("\nCheckpoint Evaluation Results:")
    for result in results_summary:
        status = "SUCCESS" if result["success"] else "FAILED"
        source_label = result.get("source", "base_path_1")
        logger.info(f"  {status} - [{source_label}]\
            Checkpoint {result['checkpoint_value']}: {result['checkpoint_path']}")

    print("\n" + "=" * 80)
    logger.info("Model Evaluation Complete")
    print("=" * 80)

    if successful_evals == 0:
        logger.error("No successful checkpoint evaluations completed")
        sys.exit(1)

    # Exit with error if any evaluations failed
    if failed_evals > 0:
        logger.info(f"\nWarning: {failed_evals} checkpoint evaluation(s) failed")
        sys.exit(0)  # Don't fail the entire job, just warn


if __name__ == "__main__":
    main()
