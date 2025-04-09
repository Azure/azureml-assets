# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for finetuning MedImageInsight."""

import argparse
import uuid
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
import faulthandler
import os
import re
import torch
import yaml
from typing import Any, Dict, List, Tuple
from safetensors.torch import save_file, load_file
from azureml.acft.image.components.mainztrain.Trainers.MainzTrainer import MainzTrainer
from azureml.acft.image.components.mainztrain.Utils.Timing import Timer
import azureml.acft.image.components.mainzvision as mv


COMPONENT_NAME = "ACFT-MedImage-Embedding-Finetune"
logger = get_logger_app("azureml.acft.contrib.hf.scripts.src.train.medimage_embedding_finetune")
CHECKPOINT_PATH = "artifacts/checkpoints/vision_model/medimageinsigt-v1.0.0.pt"
LANG_ENCODER_PATH = "artifacts/checkpoints/language_model/clip_tokenizer_4.16.2"
EVAL_IMAGE_TSV = 'EVAL_IMAGE_TSV'
EVAL_TEXT_TSV = 'EVAL_TEXT_TSV'
EVAL_TRAIN_IMAGE_TSV = 'EVAL_TRAIN_IMAGE_TSV'
EVAL_TRAIN_TEXT_TSV = 'EVAL_TRAIN_TEXT_TSV'
IMAGE_TSV = 'IMAGE_TSV'
TEXT_TSV = 'TEXT_TSV'
LABEL_FILE = 'LABEL_FILE'
SAVE_DIR = 'SAVE_DIR'
USER_DIR = 'user_dir'
MLFLOW_MODEL_FOLDER = 'MLFLOW_MODEL_FOLDER'
UNICL_MODEL = 'UNICL_MODEL'
PRETRAINED = 'PRETRAINED'
DATASET = 'DATASET'
ROOT = 'ROOT'
DATASET_ROOT = 'DATASET_ROOT'
SAMPLER = 'SAMPLER'
SET_SAMPLER_EPOCH = 'SET_SAMPLER_EPOCH'


def add_env_parser_to_yaml() -> None:
    """Adding ability of resolving environment variables to the yaml SafeLoader.

    Environment variables in the form of "${<env_var_name>}" can be resolved as strings.
    If the <env_var_name> is not in the env, <env_var_name> itself would be used.
    E.g.:
    config:
      username: admin
      password: ${SERVICE_PASSWORD}
      service: https://${SERVICE_HOST}/service
    """
    loader = yaml.SafeLoader
    env_pattern = re.compile(r".*?\${(.*?)}.*?")

    def env_constructor(loader: yaml.Loader, node: yaml.Node) -> str:
        value = loader.construct_scalar(node)
        for group in env_pattern.findall(value):
            value = value.replace(f"${{{group}}}", os.environ.get(group, group))
        return value

    yaml.add_implicit_resolver("!ENV", env_pattern, Loader=loader)
    yaml.add_constructor("!ENV", env_constructor, Loader=loader)


def load_config_dict_to_opt(opt: Dict[str, Any],
                            config_dict: Dict[str, Any],
                            splitter: str = '.',
                            log_new: bool = False) -> None:
    """Load config_dict to opt dictionary.

    Args:
        opt (Dict[str, Any]): The dictionary to be updated with values from config_dict.
        config_dict (Dict[str, Any]): The dictionary containing configuration key-value pairs to load into opt.
        splitter (str, optional): The delimiter used to split keys in config_dict. Defaults to '.'.
        log_new (bool, optional): If True, logs new keys added to opt. Defaults to False.
    Raises:
        TypeError: If config_dict is not a dictionary.
        AssertionError: If the structure of keys in config_dict does not match the expected format.
    Returns:
        None
        Load the key, value pairs from config_dict to opt, overriding existing values in opt
        if there is any.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a Python dictionary")
    for k, v in config_dict.items():
        k_parts = k.split(splitter)
        pointer = opt
        for k_part in k_parts[:-1]:
            if '[' in k_part and ']' in k_part:
                # for the format "a.b[0][1].c: d"
                k_part_splits = k_part.split('[')
                k_part = k_part_splits.pop(0)
                pointer = pointer[k_part]
                for i in k_part_splits:
                    assert i[-1] == ']'
                    pointer = pointer[int(i[:-1])]
            else:
                if k_part not in pointer:
                    pointer[k_part] = {}
                pointer = pointer[k_part]
            assert isinstance(pointer, dict), "Overriding key needs to be inside a Python dict."
        if '[' in k_parts[-1] and ']' in k_parts[-1]:
            k_part_splits = k_parts[-1].split('[')
            k_part = k_part_splits.pop(0)
            pointer = pointer[k_part]
            for i in k_part_splits[:-1]:
                assert i[-1] == ']'
                pointer = pointer[int(i[:-1])]
            assert k_part_splits[-1][-1] == ']'
            ori_value = pointer[int(k_part_splits[-1][:-1])]
            pointer[int(k_part_splits[-1][:-1])] = v
        else:
            ori_value = pointer.get(k_parts[-1])
            pointer[k_parts[-1]] = v
        if ori_value:
            logger.info(f"Overrided {k} from {ori_value} to {v}")
        elif log_new:
            logger.info(f"Added {k}: {v}")


def load_opt_from_config_files(conf_files: List[str]) -> Dict[str, Any]:
    """
    Load opt from the config files, settings in later files can override those in previous files.

    Args:
        conf_files (list): a list of config file paths

    Returns:
        dict: a dictionary of opt settings
    """
    opt = {}

    with open(conf_files, encoding='utf-8') as f:
        # config_dict = yaml.safe_load(f)
        config_dict = yaml.unsafe_load(f)

        load_config_dict_to_opt(opt, config_dict)

    return opt


def update_opt(opt):
    """Update the opt dictionary with default values and settings."""
    world_size = 1
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    if world_size == 1:
        logger.warning("=> Set 'SET_SAMPLER_EPOCH' to false when world_size is 1")
        opt['SET_SAMPLER_EPOCH'] = False

    batch_size = opt['TRAIN']['BATCH_SIZE_TOTAL']
    batch_size_per_step = opt['TRAIN']['BATCH_SIZE_PER_GPU'] * world_size
    if batch_size <= 0:
        batch_size = batch_size_per_step
        opt['TRAIN']['BATCH_SIZE_TOTAL'] = batch_size

    opt['GRADIENT_ACCUMULATE_STEP'] = batch_size // batch_size_per_step

    assert opt["GRADIENT_ACCUMULATE_STEP"] > 0, "GRADIENT_ACCUMULATE_STEP is zero"

    if 'MAX_NUM_EPOCHS' in opt:
        opt['LR_SCHEDULER_PARAMS']['epochs'] = opt['MAX_NUM_EPOCHS']

    if 'SAMPLER' in opt['DATASET'] and opt['DATASET']['SAMPLER'] == 'chunk':
        if 'TIMM_AUG' in opt['AUG']:
            logger.warning(
                '=> chunk sampler is not compatible with timm dataloader,'
                '=> update TIMM_AUG.USE_LOADER to False'
            )
            opt['AUG']['TIMM_AUG']['USE_LOADER'] = False

    if 'NAME' not in opt:
        opt['NAME'] = opt['MODEL']['NAME']


def copy_model_files(cmdline_args: Dict[str, Any]) -> None:
    """Copy all files recursively from MLFLOW_MODEL_FOLDER to MLFLOW_OUTPUT_MODEL_FOLDER.

    Also, copy the model_state_dict.pt file from the highest numbered folder in SAVE_DIR
    to the specified destination in MLFLOW_OUTPUT_MODEL_FOLDER.

    Args:
        cmdline_args (Dict[str, Any]): The command line arguments passed to the script.
    """
    import shutil

    # Copy all files recursively from MLFLOW_MODEL_FOLDER to MLFLOW_OUTPUT_MODEL_FOLDER
    src_folder = cmdline_args['MLFLOW_MODEL_FOLDER']
    dest_folder = cmdline_args['MLFLOW_OUTPUT_MODEL_FOLDER']
    logger.info(f"Copying files from {src_folder} to {dest_folder}")
    shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True)


def copy_result_files(cmdline_args: Dict[str, Any]) -> None:
    """Copy the model_state_dict.pt file from the highest numbered folder in SAVE_DIR."""
    dest_folder = cmdline_args['MLFLOW_OUTPUT_MODEL_FOLDER']

    # Find the highest numbered folder in SAVE_DIR
    save_dir = cmdline_args['SAVE_DIR']+'/INPUT_conf_files_conf~'
    run_folders = [d for d in os.listdir(save_dir)
                   if os.path.isdir(os.path.join(save_dir, d)) and d.startswith('run_')]
    if not run_folders:
        logger.error("No run folders found in SAVE_DIR")
        raise FileNotFoundError("No run folders found in SAVE_DIR")

    highest_run_folder = max(run_folders, key=lambda x: int(x.split('_')[1]))
    logger.info(f"Highest run folder found: {highest_run_folder}")
    h_run_folder_path = os.path.join(save_dir, highest_run_folder)
    logger.info(f"Highest run folder path: {h_run_folder_path}")
    logger.info(f"Searching for the highest numbered folder in {save_dir}")
    numbered_folders = \
        [d for d in os.listdir(h_run_folder_path) if os.path.isdir(os.path.join(h_run_folder_path, d)) and d.isdigit()]
    if not numbered_folders:
        logger.error("No numbered folders found in SAVE_DIR")
        raise FileNotFoundError("No numbered folders found in SAVE_DIR")

    highest_numbered_folder = max(numbered_folders, key=int)
    logger.info(f"Highest numbered folder found: {highest_numbered_folder}")
    smoothed_model_folder = os.path.join(h_run_folder_path, highest_numbered_folder, 'default', 'smoothed_model')
    model_file = os.path.join(smoothed_model_folder, 'model_state_dict.pt')

    if not os.path.exists(model_file):
        logger.error(f"model_state_dict.pt not found in {smoothed_model_folder}")
        raise FileNotFoundError(f"model_state_dict.pt not found in {smoothed_model_folder}")

    # Copy model_state_dict.pt to the specified destination
    destination_path = os.path.join(dest_folder, CHECKPOINT_PATH)
    logger.info(f"Copying model_state_dict.pt from {model_file} to {destination_path}")
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    tensor = torch.load(model_file)
    save_file(tensor, destination_path)


def get_parser() -> argparse.ArgumentParser:
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(description='Process medical images and get embeddings', allow_abbrev=False)
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The name of the task to be executed",
    )
    parser.add_argument(
        "--mlflow_model_folder",
        default="mlflow_model_folder",
        type=str,
        help="Input dir of MedImage Insight model",
    )
    parser.add_argument(
        '--eval_image_tsv',
        type=str,
        help='Path to evaluation image TSV file.'
    )
    parser.add_argument(
        '--eval_text_tsv',
        type=str,
        help='Path to evaluation text TSV file.'
    )
    parser.add_argument(
        '--eval_train_image_tsv',
        type=str,
        help='Optional path used for the evaluation task. If not specified, will use the training path.',
        default="",
        required=False
    )
    parser.add_argument(
        '--eval_train_text_tsv',
        type=str,
        help='Optional path used for the evaluation task. If not specified, will use the training path.',
        default="",
        required=False
    )
    parser.add_argument(
        '--image_tsv',
        type=str,
        help='Path to training image TSV file.'
    )
    parser.add_argument(
        '--text_tsv',
        type=str,
        help='Path to training text TSV file.'
    )
    parser.add_argument(
        '--label_file',
        type=str,
        help='Path to label file.'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        help='Directory to save the output.'
    )
    parser.add_argument(
        '--conf_files',
        type=str,
        required=True,
        help='Path(s) to the MainzTrain config file(s).'
    )
    parser.add_argument(
        '--mlflow_output_model_folder',
        type=str,
        help='Output directory for the MLflow model.'
    )

    return parser


def copy_tsv(tsv_file: str, save_dir: str) -> str:
    """Copy the TSV file to the save directory in a unique folder.

    Args:
        tsv_file (str): Path to the TSV file.
        save_dir (str): Directory to save the copied TSV file.

    Returns:
        str: The path to the copied TSV file.
    """
    unique_dir = os.path.join(save_dir, str(uuid.uuid4()))
    os.makedirs(unique_dir, exist_ok=True)
    tsv_dest = os.path.join(unique_dir, os.path.basename(tsv_file))
    os.system(f'cp {tsv_file} {tsv_dest}')
    return tsv_dest


def load_opt_command(cmdline_args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load and combine command line arguments with configuration options.

    This function processes command line arguments, loads configuration options
    from specified configuration files, and combines them into a single dictionary.
    Args:
        cmdline_args (argparse.Namespace): The command line arguments passed to the script.
    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the combined options dictionary
        and the processed command line arguments dictionary.
    """
    add_env_parser_to_yaml()
    # Extract the directory from the conf_files path
    conf_files_dir = os.path.dirname(cmdline_args.conf_files)
    conf_files = [os.path.join(conf_files_dir, f)
                  for f in os.listdir(conf_files_dir) if os.path.isfile(os.path.join(conf_files_dir, f))]
    cmdline_args.conf_files = [conf_files_dir]
    for conf_file in conf_files:
        opt = load_opt_from_config_files(conf_file)
    cmdline_args = vars(cmdline_args)
    cmdline_args = {k.upper() if k not in ('conf_files', 'world_size') else k: v for k, v in cmdline_args.items()}

    load_config_dict_to_opt(opt, cmdline_args)

    logger.info("Command line arguments:")
    for key, value in cmdline_args.items():
        logger.info(f"{key}: {value}")

    # combine cmdline_args into opt dictionary
    for key, val in cmdline_args.items():
        if val is not None:
            opt[key] = val

    # Append CHECKPOINT_PATH to mlflow_model_folder and update UNICL_MODEL's PRETRAINED key
    # Convert from safetensors to torch
    if MLFLOW_MODEL_FOLDER in cmdline_args:
        mlflow_model_path = os.path.join(cmdline_args[MLFLOW_MODEL_FOLDER], CHECKPOINT_PATH)
        safe_model = load_file(mlflow_model_path)
        os.makedirs(SAVE_DIR, exist_ok=True)
        new_path = os.path.join(SAVE_DIR, "medimageinsigt-v1.0.0-native.pt")
        torch.save(safe_model, new_path)

        opt['LANG_ENCODER']['PRETRAINED_TOKENIZER'] = os.path.join(
            cmdline_args[MLFLOW_MODEL_FOLDER], LANG_ENCODER_PATH)

        if UNICL_MODEL in opt and PRETRAINED in opt[UNICL_MODEL]:
            opt[UNICL_MODEL][PRETRAINED] = new_path

    eval_image_tsv = copy_tsv(cmdline_args[EVAL_IMAGE_TSV], opt[SAVE_DIR])
    eval_text_tsv = copy_tsv(cmdline_args[EVAL_TEXT_TSV], opt[SAVE_DIR])
    image_tsv = copy_tsv(cmdline_args[IMAGE_TSV], opt[SAVE_DIR])
    text_tsv = copy_tsv(cmdline_args[TEXT_TSV], opt[SAVE_DIR])
    label_file = copy_tsv(cmdline_args[LABEL_FILE], opt[SAVE_DIR])
    eval_train_image_tsv = image_tsv
    eval_train_text_tsv = text_tsv
    if cmdline_args.get(EVAL_TRAIN_IMAGE_TSV):
        eval_train_image_tsv = copy_tsv(cmdline_args[EVAL_TRAIN_IMAGE_TSV], opt[SAVE_DIR])
    if cmdline_args.get(EVAL_TRAIN_TEXT_TSV):
        eval_train_text_tsv = copy_tsv(cmdline_args[EVAL_TRAIN_TEXT_TSV], opt[SAVE_DIR])

    if DATASET in opt and ROOT in opt[DATASET]:
        opt[DATASET]["TRAIN_TSV_LIST"] = [image_tsv, text_tsv]
    if opt[DATASET][SAMPLER] == 'default':
        opt[SET_SAMPLER_EPOCH] = False

    for key in opt.keys():
        if key.startswith('EVALDATASET_'):
            opt[key][EVAL_IMAGE_TSV] = eval_image_tsv
            opt[key][EVAL_TEXT_TSV] = eval_text_tsv
            opt[key][IMAGE_TSV] = eval_train_image_tsv
            opt[key][TEXT_TSV] = eval_train_text_tsv
            opt[key][LABEL_FILE] = label_file

    opt[USER_DIR] = os.path.dirname(mv.__file__)
    return opt, cmdline_args


def main(args: List[str] = None) -> None:
    """Entry point for PyLearn."""
    parser = get_parser()
    args = parser.parse_args(args)
    set_logging_parameters(
        task_type=args.task_name,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
    )
    opt, _ = load_opt_command(args)
    command = 'train'
    if opt.get('SAVE_TIMER_LOG', False):
        Timer.setEnabled(True)
    logger.info('MainzTrain started')
    logger.info("Initializing distributed training options")
    if os.environ.get('MASTER_ADDR'):
        opt["MASTER_IP"] = os.environ['MASTER_ADDR']
    if os.environ.get('MASTER_PORT'):
        opt["PORT"] = os.environ['MASTER_PORT']

    logger.info(f"MASTER_IP: {opt.get('MASTER_IP')}")
    logger.info(f"PORT: {opt.get('PORT')}")
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
    logger.info(f"Rank: {rank}")
    logger.info(f"Initial opt: {opt}")
    update_opt(opt)
    logger.info(f"Updated opt: {opt}")
    trainer = MainzTrainer(opt)

    if opt.get('DEBUG_DUMP_TRACEBACKS_INTERVAL', 0) > 0:
        timeout = opt['DEBUG_DUMP_TRACEBACKS_INTERVAL']
        traceback_dir = trainer.log_folder if trainer.log_folder is not None else trainer.save_folder
        traceback_file = os.path.join(traceback_dir, f"tracebacks_{opt['rank']}.txt")
        faulthandler.dump_traceback_later(timeout, repeat=True, file=open(traceback_file, 'w'))
    if rank == 0:
        copy_model_files(opt)

    logger.info(f"Running command: {command}")
    with torch.autograd.profiler.profile(use_cuda=True,
                                         enabled=opt.get('AUTOGRAD_PROFILER', False) and opt['rank'] == 0) as prof:
        trainer.train()

    if opt.get('AUTOGRAD_PROFILER', False):
        logger.info(prof.key_averages().table(sort_by="cuda_time_total"))
        logger.info(prof.total_average())

    if opt.get('SAVE_TIMER_LOG', False):
        timer_log_dir = trainer.log_folder if trainer.log_folder is not None else trainer.save_folder
        timer_log_file = os.path.join(timer_log_dir, f"timer_log_{opt['rank']}.txt")
        Timer.timer_report(timer_log_file)

    if rank == 0:
        copy_result_files(opt)


if __name__ == "__main__":
    main()
