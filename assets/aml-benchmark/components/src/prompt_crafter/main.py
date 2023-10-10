# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Prompt Crafter Component."""
import os
from argparse import ArgumentParser
import logging

from package_3p.prompt_crafter import PromptCrafter
from utils.logging import log_mlflow_params


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=False)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--few_shot_data", type=str, required=False)
    parser.add_argument("--prompt_type", type=str, required=True, choices=['chat', 'completions'])
    parser.add_argument("--n_shots", type=int, required=True)
    parser.add_argument("--random_seed", type=int, default=0, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--output_mltable", type=str, required=False)
    parser.add_argument("--few_shot_dir", type=str, required=False)
    parser.add_argument("--input_filename", type=str, required=False)
    parser.add_argument("--few_shot_filename", type=str, required=False)
    parser.add_argument("--metadata_keys", type=str, required=False)
    parser.add_argument("--prompt_pattern", type=str, required=False)
    parser.add_argument("--few_shot_pattern", type=str, required=False)
    parser.add_argument("--few_shot_separator", type=str, required=False)
    parser.add_argument("--prefix", type=str, required=False)
    parser.add_argument("--label_map", type=str, required=False)
    parser.add_argument("--system_message", type=str, required=False)
    parser.add_argument("--output_pattern", type=str, required=False)
    parser.add_argument("--additional_payload", type=str, required=False)

    return parser.parse_args()


def main():
    args = parse_args()

    prompt_crafter = PromptCrafter(
        input_dir=args.test_data,
        few_shot_dir=args.few_shot_data,
        input_filename=None,
        few_shot_filename=None,
        prompt_type=args.prompt_type,
        n_shots=args.n_shots,
        random_seed=args.random_seed,
        output_pattern=args.output_pattern,
        prompt_pattern=args.prompt_pattern,
        few_shot_separator=args.few_shot_separator,
        prefix=args.prefix,
        output_dir=os.path.dirname(args.output_file),
        output_filename=os.path.basename(args.output_file),
        output_mltable=os.path.dirname(args.output_file),
        metadata_keys=None,
        label_map=None,
        additional_payload=None,
        system_message=args.system_message,
        few_shot_pattern=None,
    )
    prompt_crafter.run()

    log_mlflow_params(
        prompt_type=args.prompt_type,
        n_shots=args.n_shots,
        prompt_pattern=args.prompt_pattern,
        output_pattern=args.output_pattern,
        system_message=args.system_message,
        random_seed=args.random_seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
