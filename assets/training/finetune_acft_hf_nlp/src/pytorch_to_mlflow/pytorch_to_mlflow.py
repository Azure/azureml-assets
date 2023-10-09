# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import azureml.evaluate.mlflow as mlflow

from transformers import AutoTokenizer, AutoConfig


def get_parser():

    parser = argparse.ArgumentParser(description="PyTorch -> Mlflow", allow_abbrev=False)

    parser.add_argument(
        "--pytorch_model_folder",
        type=str,
        required=True,
        default=None,
        help="PyTorch Model Folder",
    )

    parser.add_argument(
        "--mlflow_model_folder",
        type=str,
        required=False,
        default="mlflow_model_folder",
        help="Mlflow Model Folder",
    )

    return parser


def main():

    parser = get_parser()
    parsed_args, _ = parser.parse_known_args()

    # load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(parsed_args.pytorch_model_folder)
    config = AutoConfig.from_pretrained(parsed_args.pytorch_model_folder)

    mlflow.hftransformers.save_model(
        parsed_args.pytorch_model_folder,
        parsed_args.mlflow_model_folder,
        tokenizer=tokenizer,
        config=config
    )


if __name__ == "__main__":
    main()
