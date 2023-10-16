# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Prompt Crafter Component."""
import os
from argparse import ArgumentParser
from typing import Optional

from .package.prompt_crafter import PromptCrafter
from .package.prompt import PromptType
from utils.exceptions import swallow_all_exceptions
from utils.logging import get_logger, log_mlflow_params


logger = get_logger(__name__)


def parse_args() -> ArgumentParser:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=False)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--few_shot_data", type=str, required=False)
    parser.add_argument("--prompt_type", type=str, required=True,
                        choices=[PromptType.chat.name, PromptType.completions.name])
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


@swallow_all_exceptions(logger)
def main(
        input_dir: str,
        prompt_type: str,
        n_shots: int,
        output_pattern: str,
        prompt_pattern: str,
        output_file: str,
        few_shot_separator: Optional[str] = None,
        prefix: Optional[str] = None,
        system_message: Optional[str] = None,
        few_shot_data: Optional[str] = None,
        random_seed: Optional[int] = 0,
) -> None:
    """Entry function for Prompt Crafter Component.

    :param input_dir: Path to jsonl would be used to generate prompts.
    :param prompt_type: Type of prompt to generate.
    :param n_shots: Number of shots to use for few-shot prompts.
    :param output_pattern: Pattern to use for output prompts.
    :param prompt_pattern: Pattern to use for prompts.
    :param output_file: Path to jsonl with generated prompts.
    :param few_shot_separator: Separator to use for few-shot prompts.
    :param prefix: Prefix to use for prompts.
    :param system_message: System message to use for prompts.
    :param few_shot_data: Path to jsonl would be used to generate n-shot prompts.
    :param random_seed: Random seed to use for prompts.
    :return: None
    """
    prompt_crafter = PromptCrafter(
        input_dir=input_dir,
        few_shot_dir=few_shot_data,
        input_filename=None,
        few_shot_filename=None,
        prompt_type=prompt_type,
        n_shots=n_shots,
        random_seed=random_seed,
        output_pattern=output_pattern,
        prompt_pattern=prompt_pattern,
        few_shot_separator=few_shot_separator,
        prefix=prefix,
        output_dir=os.path.dirname(output_file),
        output_filename=os.path.basename(output_file),
        output_mltable=os.path.dirname(args.output_file),
        metadata_keys=None,
        label_map=None,
        additional_payload=None,
        system_message=system_message,
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
    args = parse_args()

    main(
        input_dir=args.test_data,
        few_shot_dir=args.few_shot_data,
        prompt_type=args.prompt_type,
        n_shots=args.n_shots,
        random_seed=args.random_seed,
        output_pattern=args.output_pattern,
        prompt_pattern=args.prompt_pattern,
        few_shot_separator=args.few_shot_separator,
        prefix=args.prefix,
        output_file=args.output_file,
        system_message=args.system_message,
    )
