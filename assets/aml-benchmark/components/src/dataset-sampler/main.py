# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from enum import Enum
import random
import sys
import os

sys.path.append(os.getcwd())

from utils.io import resolve_io_path
from utils.helper import get_logger, log_mlflow_params


logger = get_logger(__name__)


class SamplingStyle(Enum):
    head: str = "head"
    tail: str = "tail"
    random: str = "random"


def parse_args() -> argparse.Namespace:
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
        help="Ratio of samples to be taken specified as a float in (0, 1] (alternative to --n_samples).",
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
        "--output_dataset", type=str, required=True, help="Path to the dataset output."
    )

    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


def count_lines(input_file_path: str) -> int:
    """Count the number of lines in a file

    :param input_file_path: Path to file
    :type input_file_path: str
    :return: Number of lines in file
    :rtype: int
    """
    with open(input_file_path) as f:
        n = sum(1 for _ in f)
    return n


def sample_from_head(
    input_file_path: str, output_file_path: str, n_samples: int
) -> None:
    """
    Sample from the head of the file.

    :param input_file_path: Path to file
    :type input_file_path: str
    :param output_file_path: Path to output file
    :type output_file_path: str
    :param n_samples: Absolute number of lines to sample
    :type n_samples: int
    :return: None
    :rtype: NoneType
    """
    with open(input_file_path, "r") as f, open(output_file_path, "w") as f_out:
        for i, line in enumerate(f):
            f_out.write(line)

            # stop sampling when we reach n_samples lines
            if i >= n_samples - 1:
                break


def sample_from_tail(
    input_file_path: str, output_file_path: str, line_count: int, n_samples: int
) -> None:
    """
    Sample from the tail of the file.

    :param input_file_path: Path to file
    :type input_file_path: str
    :param output_file_path: Path to output file
    :type output_file_path: str
    :param line_count: Total number of lines in file
    :type line_count: int
    :param n_samples: Absolute number of lines to sample
    :type n_samples: int
    :return: None
    :rtype: NoneType
    """
    start_index = line_count - n_samples
    with open(input_file_path, "r") as f, open(output_file_path, "w") as f_out:
        for i, line in enumerate(f):
            # start sampling when we reach start_index
            if i < start_index:
                continue

            f_out.write(line)


def sample_from_random(
    input_file_path: str,
    output_file_path: str,
    line_count: int,
    n_samples: int,
    random_seed: int,
) -> None:
    """
    Sample from the file randomly.

    :param input_file_path: Path to file
    :type input_file_path: str
    :param output_file_path: Path to output file
    :type output_file_path: str
    :param line_count: Total number of lines in file
    :type line_count: int
    :param n_samples: Absolute number of lines to sample
    :type n_samples: int
    :param random_seed: Random seed for sampling
    :type random_seed: int
    :return: None
    :rtype: NoneType
    """
    random.seed(random_seed)
    indices = set(random.sample(range(line_count), n_samples))

    logger.info(f"Using random seed: {random_seed}.")

    with open(input_file_path, "r") as f, open(output_file_path, "w") as f_out:
        for i, line in enumerate(f):
            if i in indices:
                f_out.write(line)


def main(args: argparse.Namespace) -> None:
    input_file_path = resolve_io_path(args.dataset)
    logger.info(f"Input file: {input_file_path}")
    output_file_path = args.output_dataset
    logger.info(f"Output file: {output_file_path}")

    if args.sampling_ratio is None and args.n_samples is None:
        mssg = "Either 'sampling_ratio' or 'n_samples' must be specified."
        logger.error(mssg)
        raise ValueError(mssg)
    if args.sampling_ratio is not None and args.n_samples is not None:
        mssg = "Only one of 'sampling_ratio' or 'n_samples' can be specified."
        logger.error(mssg)
        raise ValueError(mssg)

    line_count = count_lines(input_file_path)
    if args.sampling_ratio is not None:
        if not 0 < args.sampling_ratio <= 1:
            mssg = f"Invalid sampling_ratio: {args.sampling_ratio}. Please specify float in (0, 1]."
            logger.error(mssg)
            raise ValueError(mssg)
        logger.info(f"Sampling percentage: {args.sampling_ratio * 100}%.")
        n_samples = int(line_count * args.sampling_ratio)
    else:
        n_samples = args.n_samples
        if n_samples <= 0:
            mssg = f"Invalid n_samples: {n_samples}. Please specify positive integer."
            logger.error(mssg)
            raise ValueError(mssg)
        if n_samples > line_count:
            mssg = f"n_samples: {n_samples} > line_count: {line_count}. Setting n_samples = line_count."
            logger.warn(mssg)
            n_samples = line_count
    logger.info(
        f"Sampling {n_samples:,}/{line_count:,} lines using sampling_style {args.sampling_style}."
    )

    random_seed = args.random_seed
    if (
        args.sampling_style == SamplingStyle.head.value
        or args.sampling_style == SamplingStyle.tail.value
    ):
        mssg = (
            f"Received random_seed: {random_seed}, but it won't be used. It is used only when 'sampling_style' is"
            f" '{SamplingStyle.random.value}'."
        )
        logger.warn(mssg)
        random_seed = None

    if args.sampling_style == SamplingStyle.head.value:
        sample_from_head(
            input_file_path,
            output_file_path,
            n_samples=n_samples,
        )
    elif args.sampling_style == SamplingStyle.tail.value:
        sample_from_tail(
            input_file_path,
            output_file_path,
            line_count=line_count,
            n_samples=n_samples,
        )
    elif args.sampling_style == SamplingStyle.random.value:
        sample_from_random(
            input_file_path,
            output_file_path,
            line_count=line_count,
            n_samples=n_samples,
            random_seed=random_seed,
        )
    else:
        mssg = f"Invalid sampling_style: {args.sampling_style}. Please specify either 'head', 'tail' or 'random'."
        logger.error(mssg)
        raise ValueError(mssg)

    log_mlflow_params(
        input_dataset_checksum=input_file_path,
        sampling_style=args.sampling_style,
        sampling_ratio=args.sampling_ratio,
        n_samples=n_samples,
        random_seed=random_seed,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
