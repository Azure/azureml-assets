# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for data distillation component."""

import argparse

from argparse import Namespace

from pathlib import Path
from dataclasses import dataclass

from datasets.load import load_dataset
from datasets import Dataset

from distillation_adapters import ZeroShotDistillation, ChainOfDensityDistillation


@dataclass
class DataSplits:
    """Data split class."""

    TRAIN = "train"
    VALIDATION = "validation"


@dataclass
class DistillationTechniques:
    """Distillation techniques class."""

    ZERO_SHOT = "Zero-Shot"
    COD = "Chain-of-Density"


DATA_SPLIT_FILE_MAP = {
    DataSplits.TRAIN: "train_file_path",
    DataSplits.VALIDATION: "validation_file_path",
}

DATA_SPLIT_FILE_MAP = {
    DataSplits.TRAIN: "train_file_path",
    DataSplits.VALIDATION: "validation_file_path",
}

DATA_SPLIT_OUTPUT_FILE_MAP = {
    DataSplits.TRAIN: "distilled_train_dataset",
    DataSplits.VALIDATION: "distilled_validation_dataset",
}

DISTILLATION_TECHNIQUE_MAP = {
    DistillationTechniques.ZERO_SHOT: ZeroShotDistillation,
    DistillationTechniques.COD: ChainOfDensityDistillation,
}


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Task specific parameters
    parser.add_argument(
        "--text_key",
        type=str,
        help="text key",
    )

    # Distillation parameters
    parser.add_argument(
        "--distillation_technique",
        type=str,
        default=DistillationTechniques.ZERO_SHOT,
        help="distillation_technique",
    )
    parser.add_argument(
        "--cod_steps",
        type=int,
        default=4,
        help="cod_steps",
    )

    # Inputs: Dataset path parameters
    parser.add_argument(
        "--train_file_path",
        type=str,
        help="Input train file path",
    )
    parser.add_argument(
        "--validation_file_path",
        default=None,
        type=str,
        help="Input validation file path",
    )

    # Output
    parser.add_argument(
        "--distilled_train_dataset",
        type=Path,
        default=None,
        help="Output folder containing distilled train.jsonl file",
    )
    parser.add_argument(
        "--distilled_validation_dataset",
        type=Path,
        default=None,
        help="Output folder containing distilled validation.jsonl file",
    )
    parser.add_argument(
        "--distilled_test_dataset",
        type=Path,
        default=None,
        help="Output folder containing distilled test.jsonl file",
    )

    return parser


def data_distillation(args: Namespace):
    """Distill data using teacher model."""
    if args.distillation_technique in DISTILLATION_TECHNIQUE_MAP:
        distillation_processor = DISTILLATION_TECHNIQUE_MAP[
            args.distillation_technique
        ](args)
    else:
        ValueError(f"{args.distillation_technique} is not implemented.")

    for split in DATA_SPLIT_FILE_MAP:
        print(f"Processing split - {split}")

        file_name = getattr(args, DATA_SPLIT_FILE_MAP[split], None)
        if file_name is not None and Path(file_name).is_file():
            # loading data (currently support only jsonl)
            ds = load_dataset("json", data_files=file_name, split="train")
            print(ds)

            raw_data = ds[args.text_key]
            distilled_data = distillation_processor.batch_process_data(raw_data)
            print(distilled_data)

            final_data = []
            for record in distilled_data:
                print(f"Post processing record - {record['idx']}")
                new_record = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an AI assistant that summarizes text",
                        },
                        {"role": "user", "content": record["text"]},
                        {"role": "assistant", "content": record["prediction"]},
                    ]
                }
                final_data.append(new_record)
            print(final_data)

            # save final_data as jsonl file
            output_dataset = getattr(
                args,
                DATA_SPLIT_OUTPUT_FILE_MAP[split],
                None,
            )
            if output_dataset:
                Path(output_dataset).mkdir(exist_ok=True, parents=True)
                output_file = str(Path(output_dataset, f"{split}.jsonl"))
                final_ds = Dataset.from_list(final_data)
                final_ds.to_json(output_file)
        else:
            print(
                f"Skipping split {split} as {DATA_SPLIT_FILE_MAP[split]} is not provided."
            )


def main():
    """Parse args and import model."""
    # args
    parser = get_parser()
    args, _ = parser.parse_known_args()
    print(args)

    data_distillation(args)


if __name__ == "__main__":
    main()
