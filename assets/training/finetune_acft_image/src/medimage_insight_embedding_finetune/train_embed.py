import argparse
import faulthandler
import logging
import os
import sys
import torch
import yaml

from MainzTrain.Trainers.MainzTrainer import MainzTrainer
from MainzTrain.Utils.Arguments import load_opt_command
from MainzTrain.Utils.Timing import Timer

logger = logging.getLogger(__name__)

import argparse
import json
import logging
import os
import re
import yaml


def add_env_parser_to_yaml():
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

    def env_constructor(loader, node):
        value = loader.construct_scalar(node)
        for group in env_pattern.findall(value):
            value = value.replace(f"${{{group}}}", os.environ.get(group, group))
        return value

    yaml.add_implicit_resolver("!ENV", env_pattern, Loader=loader)
    yaml.add_constructor("!ENV", env_constructor, Loader=loader)


def load_config_dict_to_opt(opt, config_dict, splitter='.', log_new=False):
    """
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
            print(f"Overrided {k} from {ori_value} to {v}")
        elif log_new:
            print(f"Added {k}: {v}")


def load_opt_from_config_files(conf_files):
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




COMPONENT_NAME = "ACFT-MedImage-Embedding-Training"


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(description='Process medical images and get embeddings', allow_abbrev=False)
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
        default='/mnt/code/Users/ynagaraj/hldata/google_compare/train/effusion/',
        help='Root directory of the dataset.'
    )
    parser.add_argument(
        '--eval_image_tsv',
        type=str,
        default='/mnt/code/Users/ynagaraj/hldata/google_compare/test/effusion/ltcxr_train_cleaned/images-0.tsv',
        help='Path to evaluation image TSV file.'
    )
    parser.add_argument(
        '--eval_text_tsv',
        type=str,
        default='/mnt/code/Users/ynagaraj/hldata/google_compare/test/effusion/ltcxr_train_cleaned/text-0.tsv',
        help='Path to evaluation text TSV file.'
    )
    parser.add_argument(
        '--image_tsv',
        type=str,
        default='/mnt/code/Users/ynagaraj/hldata/google_compare/train/effusion/2k/images-0.tsv',
        help='Path to training image TSV file.'
    )
    parser.add_argument(
        '--text_tsv',
        type=str,
        default='/mnt/code/Users/ynagaraj/hldata/google_compare/train/effusion/2k/text-0.tsv',
        help='Path to training text TSV file.'
    )
    parser.add_argument(
        '--label_file',
        type=str,
        default='/mnt/code/Users/ynagaraj/hldata/google_compare/labels_effusion.txt',
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
        default='/mnt/code/Users/ynagaraj/hldata/google_compare/train/weight_effusion.txt',
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
        default='/mnt/code/Users/ynagaraj/hlldata/longtailcxr_test/images_test.zip',
        help='Path to evaluation zip file.'
    )
    parser.add_argument(
        '--eval_zip_map_file',
        type=str,
        default='/mnt/code/Users/ynagaraj/hlldata/longtailcxr_test/images_test.debug.txt',
        help='Path to evaluation zip map file.'
    )
    parser.add_argument(
        '--eval_label_file',
        type=str,
        default='/mnt/code/Users/ynagaraj/hlldata/longtailcxr_test/labels_names.txt',
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
        default='/mnt/output/finetune_google_chestxray14_effusion_ablation_04',
        help='Directory to save the output.'
    )
    parser.add_argument(
        '--conf_files',
        nargs='+',
        required=True,
        help='Path(s) to the MainzTrain config file(s).'
        )

    return parser


"""
python train_embed.py \
    --log_every 10 \
    --resume \
    --reset_data_loader \
    --fp16 \
    --zero_stage 0 \
    --deepspeed \
    --save_per_optim_steps 100 \
    --eval_per_optim_steps 100 \
    --grad_clipping 1.0 \
    --set_sampler_epoch \
    --verbose \
    --workers 6 \
    --pin_memory \
    --dataset_root /mnt/code/Users/ynagaraj/hldata/google_compare/train/effusion/ \
    --eval_image_tsv /mnt/code/Users/ynagaraj/hldata/google_compare/test/effusion/ltcxr_train_cleaned/images-0.tsv \
    --eval_text_tsv /mnt/code/Users/ynagaraj/hldata/google_compare/test/effusion/ltcxr_train_cleaned/text-0.tsv \
    --image_tsv /mnt/code/Users/ynagaraj/hldata/google_compare/train/effusion/2k/images-0.tsv \
    --text_tsv /mnt/code/Users/ynagaraj/hldata/google_compare/train/effusion/2k/text-0.tsv \
    --label_file /mnt/code/Users/ynagaraj/hldata/google_compare/labels_effusion.txt \
    --binary_metrics 1 \
    --cweight_file /mnt/code/Users/ynagaraj/hldata/google_compare/train/weight_effusion.txt \
    --zs_mode 2 \
    --zs_weight 1.0 \
    --knn 200 \
    --eval_zip_file /mnt/code/Users/ynagaraj/hlldata/longtailcxr_test/images_test.zip \
    --eval_zip_map_file /mnt/code/Users/ynagaraj/hlldata/longtailcxr_test/images_test.debug.txt \
    --eval_label_file /mnt/code/Users/ynagaraj/hlldata/longtailcxr_test/labels_names.txt \
    --batch_size_per_gpu 2 \
    --max_num_epochs 10000 \
    --gradient_accumulate_step 1 \
    --save_dir /mnt/output/finetune_google_chestxray14_effusion_ablation_04 \
    --conf_files "C:\workspace1\medimage_parse\train_yaml_versions\train.yaml"
"""


def load_opt_command(cmdline_args):
    add_env_parser_to_yaml()
    opt = load_opt_from_config_files(cmdline_args.conf_files)
    cmdline_args = vars(cmdline_args)
    cmdline_args = {k.upper() if k != 'conf_files' else k: v for k, v in cmdline_args.items()}
    
    load_config_dict_to_opt(opt, cmdline_args)
    
    print("Command line arguments:")
    for key, value in cmdline_args.items():
        print(f"{key}: {value}")
    
    # combine cmdline_args into opt dictionary
    for key, val in cmdline_args.items():
        if val is not None:
            opt[key] = val

    return opt, cmdline_args


def main(args=None):
    '''
    Main execution point for PyLearn.
    '''

    print('MainzTrain started', file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(args)
    opt, cmdline_args = load_opt_command(args)
    command = 'train'
    # if cmdline_args.user_dir:
    #     absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
    #     opt['user_dir'] = absolute_user_dir

    if opt.get('SAVE_TIMER_LOG', False):
        Timer.setEnabled(True)

    # enable attaching from PDB; use 'kill -10 PID' to enter the debugger
    def handle_pdb(sig, frame):
        import pdb
        pdb.Pdb().set_trace(frame)
    # import signal
    # signal.signal(signal.SIGUSR1, handle_pdb)
    trainer = MainzTrainer(opt)

    if opt.get('DEBUG_DUMP_TRACEBACKS_INTERVAL', 0) > 0:
        timeout = opt['DEBUG_DUMP_TRACEBACKS_INTERVAL']
        traceback_dir = trainer.log_folder if trainer.log_folder is not None else trainer.save_folder
        traceback_file = os.path.join(traceback_dir, f"tracebacks_{opt['rank']}.txt")
        faulthandler.dump_traceback_later(timeout, repeat=True, file=open(traceback_file, 'w'))

    splits = opt.get('EVALUATION_SPLITS', ["dev", "test"])

    print(f"Running command: {command}")
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
