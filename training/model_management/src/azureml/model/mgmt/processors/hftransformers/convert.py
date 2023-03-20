# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""HFTransformers convert model."""

import os
import torch
import yaml

from .config import MODEL_FILE_PATTERN
from azureml.evaluate import mlflow as hf_mlflow
from azureml.model.mgmt.processors.hftransformers.config import (
    SupportedTasks, 
    SupportedTextToImageVariants,
    SupportedNLPTasks, 
    SupportedVisionTasks,
    TaskToClassMapping,
)
from azureml.model.mgmt.utils.common_utils import copy_file_paths_to_destination, log_execution_time
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoImageProcessor
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from typing import Dict


def _get_default_task_signatures(task_type) -> Dict:
    """Return mlflow i/p and o/p signature for a hftransformers supported task."""
    if task_type == SupportedTasks.TEXT_TO_IMAGE.value:
        return {
            "inputs": '[{"name": "input_string", "type": "string"}]',
            "outputs": '[{"name": "image", "type": "string"}]'
        }
    elif SupportedVisionTasks.has_value(task_type):
        return {
            "inputs": '[{"name": "image", "type": "string"}]',
            "outputs": '[{"name": "probs", "type": "string"}, {"name": "labels", "type": "string"}]'
        }
    elif SupportedNLPTasks.has_value(task_type):
        if task_type == SupportedTasks.QUESTION_ANSWERING.value:
            return {
                "inputs": '[{"name": "question", "type": "string"}, {"name": "context", "type": "string"}]',
                "outputs": '[{"type": "string"}]'
            }
        # For all the supported tasks
        return {
            "inputs": '[{"name": "input_string", "type": "string"}]',
            "outputs": '[{"type": "string"}]'
        }
    return {}


def _add_mlflow_signature(mlflow_model_path: Path, signature):
    print("Adding mlflow signatures")
    mlmodel_path = mlflow_model_path / "MLmodel"
    updated_yaml_dict = {}
    # read YAML file
    with open(mlmodel_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
        updated_yaml_dict.update(yaml_dict)
        updated_yaml_dict["signature"] = signature
    # save updated values to MLModel file
    with open(mlmodel_path, "w") as f:
        yaml.dump(updated_yaml_dict, f)


def _get_image_model_to_save(input_dir: Path, output_dir: Path, hf_conf: Dict = {}) -> Dict:
    """Save Huggingface image models to mlflow and return hftransformers accepted parameters."""
    temp_output_dir = Path(output_dir).parent.absolute() / "tmp"
    model_dir = temp_output_dir / "model"
    copy_file_paths_to_destination(input_dir, model_dir, MODEL_FILE_PATTERN)

    config = AutoConfig.from_pretrained(input_dir, local_files_only=True)
    image_processor = AutoImageProcessor.from_pretrained(input_dir, config=config, local_files_only=True)

    hf_conf["hf_predict_module"] = "hf_test_predict"
    hf_conf["hf_tokenizer_class"] = AutoImageProcessor.__name__
    hf_conf["train_label_list"] = sorted(list(config.label2id.keys()))

    predict_script = os.path.join(os.path.dirname(__file__), "image_export_assets", "hf_test_predict.py")
    requirements_file = os.path.join(os.path.dirname(__file__), "image_export_assets", "requirements.txt")

    return {
        "hf_model": str(model_dir),
        "path": output_dir,
        "tokenizer": image_processor,
        "config": config,
        "hf_conf": hf_conf,
        "code_paths": [predict_script],
        "pip_requirements": requirements_file
    }


def _get_stable_difussion_model_to_save(input_dir: Path, output_dir: Path, hf_conf: Dict = {}) -> Dict:
    """Save Huggingface stable diffusion model to mlflow and return hftransformers accepted parameters."""
    model = None
    try:
        print("Cuda available" if torch.cuda.is_available() else "Cuda not available")
        if torch.cuda.is_available():  # correct?
            device = "cuda"
            pipe = StableDiffusionPipeline.from_pretrained(input_dir, local_files_only=True, torch_dtype=torch.float16)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            model = pipe.to(device)
    except Exception as e:
        print("Error in loading stable diffusion model")

    if not model:    
        model = StableDiffusionPipeline.from_pretrained(input_dir, local_files_only=True, torch_dtype=torch.float16)

    predict = os.path.join(os.path.dirname(__file__), "diffusion", "predict.py")
    hf_conf['custom_config_module'] = "diffusers"
    hf_conf['custom_tokenizer_module'] = "diffusers"
    hf_conf['custom_model_module'] = "diffusers"
    return {
        "hf_model": model,
        "hf_conf": hf_conf,
        "path": output_dir,
        "code_paths": [predict]
    }


def _get_nlp_model_to_save(input_dir: Path, output_dir: Path, hf_conf: Dict = {}) -> Dict:
    """Save Huggingface NLP model to mlflow and return hftransformers accepted parameters."""
    # prepare model files in expected format
    temp_output_dir = Path(output_dir).parent.absolute() / "tmp"
    model_dir = temp_output_dir / "model"
    copy_file_paths_to_destination(input_dir, model_dir, MODEL_FILE_PATTERN)

    config = AutoConfig.from_pretrained(input_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(input_dir, config=config, local_files_only=True)

    return {
        "hf_model": str(model_dir),
        "tokenizer": tokenizer,
        "config": config,
        "path": output_dir,
        "hf_conf": hf_conf,
    }


@log_execution_time
def to_mlflow(input_dir: Path, output_dir: Path, translate_params: Dict):
    """Convert Hugging face pytorch model to Mlflow."""
    signatures = translate_params.get('signature')
    model_id = translate_params['model_id']
    task_type = translate_params['task_type']

    task_category = task_type if "stable-diffusion" not in model_id else "stable-diffusion"
    hf_pretrained_class = TaskToClassMapping.get_automodel_class_name(task_category)

    hf_conf = {
        'task_type': task_type,
        'hf_pretrained_class': hf_pretrained_class,
        'huggingface_id': model_id,
    }

    if SupportedTextToImageVariants.has_value(task_category):
        model_configs = _get_stable_difussion_model_to_save(input_dir, output_dir, hf_conf)
    elif SupportedVisionTasks.has_value(task_category):
        model_configs = _get_nlp_model_to_save(input_dir, output_dir, hf_conf)
    elif SupportedVisionTasks.has_value(task_category):
        model_configs = _get_image_model_to_save(input_dir, output_dir, hf_conf)
    else:
        raise Exception("Unsupported model or task type")

    print(f"Saving hftranformers model to path {output_dir}")
    print(f"hftransformers parameters: \n{model_configs}\n")
    hf_mlflow.hftransformers.save_model(**model_configs)

    # add signatures
    signatures = signatures if signatures else _get_default_task_signatures(task_type)
    _add_mlflow_signature(output_dir, signatures)
    print("Model saved!!!")
