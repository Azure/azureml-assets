# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Prompt Crafter Component."""
from argparse import ArgumentParser
from typing import Optional

from .package.prompt_crafter import PromptCrafter
from .package.prompt import PromptType
from aml_benchmark.utils.exceptions import swallow_all_exceptions
from aml_benchmark.utils.logging import get_logger, log_mlflow_params
from aml_benchmark.utils.io import resolve_io_path

logger = get_logger(__name__)


def parse_args() -> ArgumentParser:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to load the test data to generate prompts."
        )
    parser.add_argument(
        "--few_shot_data",
        type=str,
        required=False,
        help="The uri file(jsonl) to generate n-shot prompts.")
    parser.add_argument(
        "--prompt_type",
        type=str,
        required=True,
        choices=[PromptType.chat.name, PromptType.completions.name],
        help="The type of prompt to be generated.")
    parser.add_argument(
        "--n_shots",
        type=int,
        required=True,
        help="The number of shots to use for n-shot prompts.")
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        required=False,
        help="The random seed to used for generating prompts.")
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output jsonl which has the generated prompts.")
    parser.add_argument(
        "--prompt_pattern",
        type=str,
        required=True,
        help="The jinja pattern to used for generating prompts.")
    parser.add_argument(
        "--few_shot_separator",
        type=str,
        default="",
        required=False,
        help="The separator to be used for n-shot prompts.")
    parser.add_argument(
        "--prefix",
        type=str,
        required=False,
        help="The prefix to be used for prompts.")
    parser.add_argument(
        "--label_map",
        type=str,
        required=False,
        help="The label map to be used for prompts.")
    parser.add_argument(
        "--ground_truth_column_name",
        type=str,
        required=False,
        help="The ground truth key in the input data.")
    parser.add_argument(
        "--system_message",
        type=str,
        required=False,
        help="The system message to be used for chat prompts.")
    parser.add_argument(
        "--output_pattern",
        type=str,
        required=True,
        help="The jinja template representing the expected output to \
         be used for few shot prompts when n_shot > 0")
    return parser.parse_args()


@swallow_all_exceptions(logger)
def main(
        test_data: str,
        prompt_type: str,
        n_shots: int,
        output_pattern: str,
        prompt_pattern: str,
        output_file: str,
        ground_truth_column_name: Optional[str] = None,
        few_shot_separator: Optional[str] = None,
        prefix: Optional[str] = None,
        system_message: Optional[str] = None,
        few_shot_data: Optional[str] = None,
        random_seed: Optional[int] = 0,
) -> None:
    """Entry function for Prompt Crafter Component.

    :param test_data: Path to jsonl would be used to generate prompts.
    :param prompt_type: Type of prompt to generate.
    :param n_shots: Number of shots to use for few-shot prompts.
    :param output_pattern: Pattern to use for output prompts.
    :param prompt_pattern: Pattern to use for prompts.
    :param output_file: Path to jsonl with generated prompts.
    :param ground_truth_column_name: Ground truth column name.
    :param few_shot_separator: Separator to use for few-shot prompts.
    :param prefix: Prefix to use for prompts.
    :param system_message: System message to use for prompts.
    :param few_shot_data: Path to jsonl would be used to generate n-shot prompts.
    :param random_seed: Random seed to use for prompts.
    :return: None
    """
    prompt_crafter = PromptCrafter(
        test_data=test_data,
        few_shot_data=few_shot_data,
        prompt_type=prompt_type,
        n_shots=n_shots,
        random_seed=random_seed,
        output_pattern=output_pattern,
        prompt_pattern=prompt_pattern,
        few_shot_separator=few_shot_separator,
        prefix=prefix,
        output_file=output_file,
        ground_truth_column_name=ground_truth_column_name,
        output_mltable=None,
        metadata_keys=None,
        label_map=None,
        additional_payload=None,
        system_message=system_message,
        few_shot_pattern=None,
    )
    prompt_crafter.run()

    log_mlflow_params(
        prompt_type=prompt_type,
        n_shots=n_shots,
        prompt_pattern=prompt_pattern,
        output_pattern=output_pattern,
        system_message=system_message,
        random_seed=random_seed,
        ground_truth_column_name=ground_truth_column_name
        if ground_truth_column_name else None,
        test_dataset_checksum=resolve_io_path(test_data),
        few_shot_dataset_checksum=resolve_io_path(few_shot_data)
        if few_shot_data else None,
        output_dataset_checksum=resolve_io_path(output_file))


if __name__ == "__main__":
    args = parse_args()

    main(
        test_data=args.test_data,
        few_shot_data=args.few_shot_data,
        prompt_type=args.prompt_type,
        n_shots=args.n_shots,
        random_seed=args.random_seed,
        output_pattern=args.output_pattern,
        prompt_pattern=args.prompt_pattern,
        few_shot_separator=args.few_shot_separator,
        prefix=args.prefix,
        ground_truth_column_name=args.ground_truth_column_name,
        output_file=args.output_file,
        system_message=args.system_message,
    )
