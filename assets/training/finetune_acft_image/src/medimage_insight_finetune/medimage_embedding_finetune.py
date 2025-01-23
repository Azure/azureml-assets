import argparse
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
import faulthandler
import os
import re
import sys
import torch
import yaml
from typing import Any, Dict, List, Tuple

from MainzTrain.Trainers.MainzTrainer import MainzTrainer
from MainzTrain.Utils.Timing import Timer

COMPONENT_NAME = "ACFT-MedImage-Embedding-Finetune"
logger = get_logger_app("azureml.acft.contrib.hf.scripts.src.train.medimage_embedding_finetune")


def add_env_parser_to_yaml() -> None:
    """
    Adding ability of resolving environment variables to the yaml SafeLoader.
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


def load_config_dict_to_opt(opt: Dict[str, Any], config_dict: Dict[str, Any], splitter: str = '.', log_new: bool = False) -> None:
    """
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
    for conf_file in conf_files:
        with open(conf_file, encoding='utf-8') as f:
            # config_dict = yaml.safe_load(f)
            config_dict = yaml.unsafe_load(f)

        load_config_dict_to_opt(opt, config_dict)

    return opt


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
        '--log_every',
        type=int,
        default=10,
        help='Log every n steps.'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint.'
    )
    parser.add_argument(
        '--reset_data_loader',
        action='store_false',
        help='Reset data loader.'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use FP16 precision.'
    )
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage.'
    )
    parser.add_argument(
        '--deepspeed',
        action='store_false',
        help='Use DeepSpeed optimization.'
    )
    parser.add_argument(
        '--save_per_optim_steps',
        type=int,
        default=100,
        help='Save checkpoint every n optimization steps.'
    )
    parser.add_argument(
        '--eval_per_optim_steps',
        type=int,
        default=100,
        help='Evaluate every n optimization steps.'
    )
    parser.add_argument(
        '--grad_clipping',
        type=float,
        default=1.0,
        help='Gradient clipping value.'
    )
    parser.add_argument(
        '--set_sampler_epoch',
        action='store_false',
        help='Set sampler epoch.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=6,
        help='Number of workers.'
    )
    parser.add_argument(
        '--pin_memory',
        action='store_true',
        help='Pin memory in data loader.'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        help='Root directory of the dataset.'
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
        '--binary_metrics',
        type=int,
        default=1,
        help='Use binary metrics.'
    )
    parser.add_argument(
        '--cweight_file',
        type=str,
        help='Path to class weight file.'
    )
    parser.add_argument(
        '--zs_mode',
        type=int,
        default=2,
        help='Zero-shot mode.'
    )
    parser.add_argument(
        '--zs_weight',
        type=float,
        default=1.0,
        help='Zero-shot weight.'
    )
    parser.add_argument(
        '--knn',
        type=int,
        default=200,
        help='Number of nearest neighbors for KNN.'
    )
    parser.add_argument(
        '--eval_zip_file',
        type=str,
        help='Path to evaluation zip file.'
    )
    parser.add_argument(
        '--eval_zip_map_file',
        type=str,
        help='Path to evaluation zip map file.'
    )
    parser.add_argument(
        '--eval_label_file',
        type=str,
        help='Path to evaluation label file.'
    )
    parser.add_argument(
        '--batch_size_per_gpu',
        type=int,
        default=2,
        help='Batch size per GPU.'
    )
    parser.add_argument(
        '--max_num_epochs',
        type=int,
        default=10000,
        help='Maximum number of epochs.'
    )
    parser.add_argument(
        '--gradient_accumulate_step',
        type=int,
        default=1,
        help='Number of gradient accumulation steps.'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        help='Directory to save the output.'
    )
    parser.add_argument(
        '--conf_files',
        nargs='+',
        required=True,
        help='Path(s) to the MainzTrain config file(s).'
        )

    return parser


def load_opt_command(cmdline_args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load and combine command line arguments with configuration options.
    This function processes command line arguments, loads configuration options
    from specified configuration files, and combines them into a single dictionary.
    Args:
        cmdline_args (argparse.Namespace): The command line arguments passed to the script.
    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the combined options dictionary
        and the processed command line arguments dictionary.
    """
    add_env_parser_to_yaml()
    opt = load_opt_from_config_files(cmdline_args.conf_files)
    cmdline_args = vars(cmdline_args)
    cmdline_args = {k.upper() if k != 'conf_files' else k: v for k, v in cmdline_args.items()}

    load_config_dict_to_opt(opt, cmdline_args)

    logger.info("Command line arguments:")
    for key, value in cmdline_args.items():
        logger.info(f"{key}: {value}")

    # combine cmdline_args into opt dictionary
    for key, val in cmdline_args.items():
        if val is not None:
            opt[key] = val

    return opt, cmdline_args


def main(args: List[str] = None) -> None:
    '''
    Main execution point for PyLearn.
    '''

    logger.info('MainzTrain started')
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

    trainer = MainzTrainer(opt)

    if opt.get('DEBUG_DUMP_TRACEBACKS_INTERVAL', 0) > 0:
        timeout = opt['DEBUG_DUMP_TRACEBACKS_INTERVAL']
        traceback_dir = trainer.log_folder if trainer.log_folder is not None else trainer.save_folder
        traceback_file = os.path.join(traceback_dir, f"tracebacks_{opt['rank']}.txt")
        faulthandler.dump_traceback_later(timeout, repeat=True, file=open(traceback_file, 'w'))

    splits = opt.get('EVALUATION_SPLITS', ["dev", "test"])

    logger.info(f"Running command: {command}")
    with torch.autograd.profiler.profile(use_cuda=True, enabled=opt.get('AUTOGRAD_PROFILER', False) and opt['rank'] == 0) as prof:
        if command == "train":
            trainer.train()
        elif command == "evaluate":
            trainer.eval(splits=splits)
        elif command == 'train-and-evaluate':
            best_checkpoint_path = trainer.train()
            opt['PYLEARN_MODEL'] = best_checkpoint_path
            trainer.eval(splits=splits)
        else:
            raise ValueError(f"Unknown command: {command}")

    if opt.get('AUTOGRAD_PROFILER', False):
        logger.info(prof.key_averages().table(sort_by="cuda_time_total"))
        logger.info(prof.total_average())

    if opt.get('SAVE_TIMER_LOG', False):
        timer_log_dir = trainer.log_folder if trainer.log_folder is not None else trainer.save_folder
        timer_log_file = os.path.join(timer_log_dir, f"timer_log_{opt['rank']}.txt")
        Timer.timer_report(timer_log_file)


if __name__ == "__main__":
    main()
