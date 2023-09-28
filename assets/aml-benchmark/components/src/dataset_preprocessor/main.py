# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Entry script for Dataset Preprocessor Component."""

import argparse
import os

from utils.logging import get_logger, log_mlflow_params
from utils.exceptions import swallow_all_exceptions

from . import dataset_preprocessor as dsp

logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default=os.getcwd(),
        help="Path to load the input dataset"
    )
    parser.add_argument(
        "--template_input",
        type=str,
        default=None,
        help="A json dictionary where key is the name of the column enclosed in " " and associated \
            dict value is presented using jinja template logic which will be used to extract the \
            respective value from the dataset."
    )
    parser.add_argument(
        "--script_path",
        type=str,
        default=None,
        help="Path to the custom preprocessor python script provided by user."
    )
    parser.add_argument(
        "--encoder_config",
        type=str,
        default=None,
        help=('JSON serialized dictionary to perform mapping. Must contain key-value pair'
              '"column_name": "<actual_column_name>" whose value needs mapping, followed by'
              'key-value pairs containing idtolabel or labeltoid mappers.'
              'Example: {"column_name":"label", "0":"NEUTRAL", "1":"ENTAILMENT", "2":"CONTRADICTION"}.'
              'This is not applicable to custom scripts.')
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        help="Path to the jsonl file where the processed data will be saved."
    )
    argss, _ = parser.parse_known_args()
    return argss


@swallow_all_exceptions(logger)
def main(
    dataset: str,
    template_input: str,
    script_path: str,
    encoder_config: str,
    output_dataset: str
) -> None:
    """
    Entry function for Dataset Preprocessor.

    :param dataset: Path to the jsonl file to load the dataset.
    :param template_input: A json dictionary where key is the name of the column enclosed in " " and associated \
        dict value is presented using jinja template logic which will be used to extract the \
        respective value from the dataset.
    :param script_path: Path to the custom preprocessor python script provided by user.
    :param encoder_config: JSON serialized dictionary to perform mapping. Must contain key-value pair \
        "column_name": "<actual_column_name>" whose value needs mapping, followed by key-value pairs containing \
        idtolabel or labeltoid mappers. This is not aplicable to custom scripts.
    :param output_dataset: Path to the jsonl file where the processed data will be saved.
    :return: None
    """
    processor = dsp.DatasetPreprocessor(
        input_dataset=dataset,
        template=template_input,
        user_preprocessor=script_path,
        encoder_config=encoder_config,
        output_dataset=output_dataset
    )
    processor.run()
    log_mlflow_params(
        dataset=dataset,
        template_input=template_input if template_input else None,
        script_path=script_path if script_path else None,
        encoder_config=encoder_config if encoder_config else None,
        output_dataset=output_dataset
    )
    return


if __name__ == "__main__":
    argss = parse_arguments()
    main(
        dataset=argss.dataset,
        template_input=argss.template_input,
        script_path=argss.script_path,
        encoder_config=argss.encoder_config,
        output_dataset=argss.output_dataset
    )
