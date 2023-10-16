# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Dataset Sampler Component."""

from typing import List, Literal, Optional
import argparse
from enum import Enum
import random

from azureml._common._error_definition.azureml_error import AzureMLError

from utils.io import resolve_io_path
from utils.logging import get_logger, log_mlflow_params
from utils.exceptions import swallow_all_exceptions, BenchmarkValidationException
from utils.error_definitions import BenchmarkValidationError


logger = get_logger(__name__)


class SamplingStyle(Enum):
    """Enum for sampling style."""

    head: str = "head"
    tail: str = "tail"
    random: str = "random"
    duplicate: str = "duplicate"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the input .jsonl file from which the data will be sampled.",
    )
    parser.add_argument(
        "--sampling_style",
        type=str,
        required=True,
        choices=[member.value for member in SamplingStyle],
        help="Strategy used to sample. Either 'head', 'tail' or 'random'.",
    )
    parser.add_argument(
        "--sampling_ratio",
        type=float,
        default=None,
        help=(
            "Ratio of samples to be taken specified as a float > 0 (alternative to --n_samples). "
            "NOTE: sampling_ratio > 1 is only OK for `duplicate` sampling style.",
        )
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Absolute number of samples to be taken (alternative to 'sampling_ratio').",
    )
    parser.add_argument(
        "--random_seed", type=int, default=0, help="Random seed for sampling mode."
    )
    parser.add_argument(
        "--output_dataset", type=str, required=True, help="Path to the output file."
    )

    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


def count_lines(input_file_paths: List[str]) -> List[int]:
    """Return the list containing number of lines in a list of files.

    :param input_file_paths: List of file paths
    :return: List containing number of lines in the list of files
    """
    line_counts = []
    for input_file_path in input_file_paths:
        with open(input_file_path) as f:
            n = sum(1 for _ in f)
            line_counts.append(n)
    return line_counts


def sample_from_head(
    input_file_paths: List[str], output_path: str, n_samples: List[int]
) -> None:
    """
    Sample from the head of the file.

    :param input_file_paths: List of file paths
    :param output_path: Path to output file
    :param n_samples: List of absolute number of lines to sample
    :return: None
    """
    for i, (input_file_path, n_sample) in enumerate(zip(input_file_paths, n_samples)):
        with open(input_file_path, "r") as f, open(output_path, "a") as f_out:
            for i, line in enumerate(f):
                f_out.write(line)

                # stop sampling when we reach n_sample lines
                if i >= n_sample - 1:
                    break


def sample_from_tail(
    input_file_paths: List[str], output_path: str, line_counts: List[int], n_samples: List[int]
) -> None:
    """
    Sample from the tail of the file.

    :param input_file_paths: List of file paths
    :param output_path: Path to output file
    :param line_counts: List of number of lines in the list of files
    :param n_samples: List of absolute number of lines to sample
    :return: None
    """
    for i, (input_file_path, line_count, n_sample) in enumerate(zip(input_file_paths, line_counts, n_samples)):
        start_index = line_count - n_sample
        with open(input_file_path, "r") as f, open(output_path, "a") as f_out:
            for i, line in enumerate(f):
                # start sampling when we reach start_index
                if i < start_index:
                    continue

                f_out.write(line)


def sample_from_random(
    input_file_paths: List[str],
    output_path: str,
    line_counts: List[int],
    n_samples: List[int],
    random_seed: int,
) -> None:
    """
    Sample from the file randomly.

    :param input_file_paths: List of file paths
    :param output_path: Path to output file
    :param line_counts: List of number of lines in the list of files
    :param n_samples: List of absolute number of lines to sample
    :param random_seed: Random seed for sampling
    :return: None
    """
    random.seed(random_seed)
    logger.info(f"Using random seed: {random_seed}.")
    for i, (input_file_path, line_count, n_sample) in enumerate(zip(input_file_paths, line_counts, n_samples)):
        indices = set(random.sample(range(line_count), n_sample))
        with open(input_file_path, "r") as f, open(output_path, "a") as f_out:
            for i, line in enumerate(f):
                if i in indices:
                    f_out.write(line)


def sample_from_duplicate(
    input_file_paths: List[str], output_path: str, n_samples: List[int]
) -> None:
    """
    Sample by duplicating the rows of the file.

    :param input_file_paths: List of file paths
    :param output_path: Path to output file
    :param n_samples: List of absolute number of lines to sample
    :return: None
    """
    for i, (input_file_path, n_sample) in enumerate(zip(input_file_paths, n_samples)):
        counter = 0
        with open(input_file_path, "r") as f, open(output_path, "a") as f_out:
            while counter < n_sample:
                f.seek(0)
                for line in f:
                    counter += 1
                    f_out.write(line)

                    # stop sampling when we reach n_sample lines
                    if counter >= n_sample:
                        break


@swallow_all_exceptions(logger)
def main(
    output_dataset: str,
    dataset: str,
    sampling_style: Literal[SamplingStyle.head, SamplingStyle.tail, SamplingStyle.random],
    sampling_ratio: Optional[float] = None,
    n_samples: Optional[int] = None,
    random_seed: int = 0,
) -> None:
    """
    Entry function for Dataset Sampler Component.

    Either `sampling_ratio` or `n_samples` must be supplied; but not both.

    :param output_dataset: Path to the file where the sampled dataset will be saved.
    :param dataset: Path to the input directory or .jsonl file from which the data will be sampled.
    :param sampling_style: Strategy used to sample. Either 'head', 'tail' or 'random'.
    :param sampling_ratio: Ratio of samples to be taken specified as a float in (0, 1] (alternative to `n_samples`).
    :param n_samples: Absolute number of samples to be taken (alternative to `sampling_ratio`).
    :param random_seed: Random seed for sampling mode; if not specified, 0 is used. Used only when
    `sampling_style` is `random`.
    :return: None
    """
    input_file_paths = [file for file in resolve_io_path(dataset) if file.endswith(".jsonl")]
    if not input_file_paths:
        mssg = f"No .jsonl files found in {dataset}."
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details=mssg)
        )
    logger.info(f"Input file: {input_file_paths}")
    logger.info(f"Output path: {output_dataset}")

    if sampling_ratio is None and n_samples is None:
        mssg = "Either 'sampling_ratio' or 'n_samples' must be specified."
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details=mssg)
        )
    if sampling_ratio is not None and n_samples is not None:
        mssg = "Only one of 'sampling_ratio' or 'n_samples' can be specified."
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details=mssg)
        )

    line_counts = count_lines(input_file_paths)
    if sampling_ratio is not None:
        if not 0 < sampling_ratio:
            mssg = f"Invalid sampling_ratio: {sampling_ratio}. Please specify a positive float number."
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
        if sampling_ratio > 1 and sampling_style != SamplingStyle.duplicate.value:
            mssg = (
                f"Invalid sampling_ratio: sampling_ratio > 1 only allowed for sampling style `duplicate`. "
                f"Received: sampling_style={sampling_style}."
            )
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
        logger.info(f"Sampling percentage: {sampling_ratio * 100}%.")
        n_samples = [int(line_count * sampling_ratio) for line_count in line_counts]
    else:
        if n_samples <= 0:
            mssg = f"Invalid n_samples: {n_samples}. Please specify positive integer."
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
        n_samples = [n_samples] * len(line_counts)
        if any(n_sample > line_count for n_sample, line_count in zip(n_samples, line_counts)):
            mssg = (
                f"At least one member in n_samples: {n_samples} > line_counts: {line_counts}. "
                f"Setting n_samples = {min(line_counts)}."
            )
            logger.warn(mssg)
            n_samples = [min(line_counts)] * len(line_counts)
    logger.info(
        f"Sampling {n_samples}/{line_counts} lines using sampling_style {sampling_style}."
    )

    if (
        sampling_style == SamplingStyle.head.value
        or sampling_style == SamplingStyle.tail.value
    ):
        mssg = (
            f"Received random_seed: {random_seed}, but it won't be used. It is used only when 'sampling_style' is"
            f" '{SamplingStyle.random.value}'."
        )
        logger.warn(mssg)
        random_seed = None

    if sampling_style == SamplingStyle.head.value:
        sample_from_head(
            input_file_paths,
            output_dataset,
            n_samples=n_samples,
        )
    elif sampling_style == SamplingStyle.tail.value:
        sample_from_tail(
            input_file_paths,
            output_dataset,
            line_counts=line_counts,
            n_samples=n_samples,
        )
    elif sampling_style == SamplingStyle.random.value:
        sample_from_random(
            input_file_paths,
            output_dataset,
            line_counts=line_counts,
            n_samples=n_samples,
            random_seed=random_seed,
        )
    elif sampling_style == SamplingStyle.duplicate.value:
        sample_from_duplicate(
            input_file_paths,
            output_dataset,
            n_samples=n_samples,
        )
    else:
        mssg = f"Invalid sampling_style: {sampling_style}. Please specify either 'head', 'tail' or 'random'."
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details=mssg)
        )

    log_mlflow_params(
        dataset=input_file_paths,
        sampling_style=sampling_style,
        sampling_ratio=sampling_ratio,
        n_samples=n_samples[0] if sampling_ratio is None else None,
        random_seed=random_seed if sampling_style == SamplingStyle.random.value else None,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset=args.dataset,
        sampling_style=args.sampling_style,
        sampling_ratio=args.sampling_ratio,
        n_samples=args.n_samples,
        random_seed=args.random_seed,
        output_dataset=args.output_dataset,
    )
