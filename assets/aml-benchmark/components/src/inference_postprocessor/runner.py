# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Entry script for 3P Inference Postprocessor Component."""

import argparse
import os
from utils.logging import get_logger, log_mlflow_params
from utils.exceptions import swallow_all_exceptions

from . import inference_postprocessor as inferpp

logger = get_logger(__name__)
OUTPUT_FILENAME = "extracted_data.jsonl"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to dir to load the prediction dataset.",
    )
    parser.add_argument(
        "--input_uri_file",
        type=str,
        help="Path to jsonl file to load the prediction dataset.",
    )
    parser.add_argument(
        "--input_filename",
        type=str,
        default='few_shot_prompt',
        help="Name of the prediction dataset.",
    )
    parser.add_argument(
        "--prediction_column_name",
        type=str,
        help="Prediction column name.",
    )
    parser.add_argument(
        "--completion_key",
        type=str,
        required=True,
        default=None,
        help="The key in the jsonl file that contains the inference results..",
    )
    parser.add_argument(
        "--extracted_prediction_key",
        type=str,
        default='prediction',
        help="The key in the jsonl file that will contain the extracted inference results.",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=None,
        help="Few shot separator used in prompt crafter.",
    )
    parser.add_argument(
        "--find_first",
        type=str,
        default=None,
        help=(
            "A list of strings to search for in the inference results."
            "The first occurrence of each string will be extracted."
            "Must provide a comma-separated list of strings."
        ),
    )
    parser.add_argument(
        "--regex",
        type=str,
        default=None,
        help=(
            "A regular expression to extract the answer from the inference results."
            "The pattern must contain a group to be extracted. The first group and the"
            "first match will be used."
        ),
    )
    parser.add_argument(
        "--remove_prefixes",
        type=str,
        default=None,
        help=(
            "A set of string prefixes separated by comma list of string prefixes to be removed"
            "from the inference results in sequence. This can also be used to remove the prompt"
            "from the inference results. The prefixes should be separated by a comma."
        ),
    )
    parser.add_argument(
        "--remove_prompt_prefix",
        type=bool,
        default=None,
        help=(
            'If true, remove the value in the "prompt" key from the start of the completion key.'
            'Typical use case is to strip out input strings that are repeated in the output.'
            'Note this is the first filtering or modification that is applied.'
        ),
    )
    parser.add_argument(
        "--strip_characters",
        type=str,
        default=None,
        help=(
            "A set of characters to remove from the beginning or end of the extracted answer."
            "It is applied in the very end of the extraction process."
        ),
    )
    parser.add_argument(
        "--extract_number",
        type=str,
        default=None,
        help=(
            "If the inference results contain a number, this can be used to extract the first or last"
            "number in the inference results. The number will be extracted as a string."
        ),
    )
    parser.add_argument(
        "--label_map",
        type=str,
        default=None,
        help=(
            "JSON serialized dictionary to perform mapping. Must contain key-value pair"
            '"column_name": "<actual_column_name>" whose value needs mapping, followed by'
            "key-value pairs containing idtolabel or labeltoid mappers."
            'Example: {"column_name":"label", "0":"NEUTRAL", "1":"ENTAILMENT", "2":"CONTRADICTION"}.'
            "This is not applicable to custom scripts."
        ),
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        default=None,
        help="Path to the jsonl file where the processed data will be saved.",
    )
    argss, _ = parser.parse_known_args()
    return argss


@swallow_all_exceptions(logger)
def main(
    prediction_dir: str = None,
    prediction_filename: str = None,
    prediction_dataset: str = None,
    prediction_column_name: str = None,
    completion_key: str = None,
    separator: str = None,
    find_first: str = None,
    regex_expr: str = None,
    remove_prefixes: str = None,
    strip_characters: str = None,
    extract_number: str = None,
    remove_prompt_prefix: str = None,
    label_map: str = None,
    output_dataset: str = None,
) -> None:
    """
    Entry function for Inference Postprocessor.

    :param prediction_dir: Path to the directory containing the jsonl file with the inference results. If \
        prediction_dataset is specified, prediction_dataset takes priority.
    :param prediction_filename: The name of the jsonl file with the inference results. If \
        prediction_dataset is specified, prediction_dataset takes priority.
        The name of the jsonl file with the inference results. Supports any glob pattern that returns a \
        unique .jsonl file within the specified directory. Gets ignored if prediction_dataset is specified.
    :param prediction_dataset: Path to the jsonl file to load the prediction dataset.
    :param prediction_column_name: Name of prediction column/key.
    :param ground_truth_dataset: Path to the jsonl file to load the ground truth dataset.
    :param ground_truth_column_name: Name of ground truth column/key.
    :param separator: Few shot separator used in prompt crafter.
    :param find_first: A list of strings to search for in the inference results. The first occurrence \
        of each string will be extracted. Must provide a comma-separated list of strings.
    :param regex_expr: A regex pattern to extract the answer from the inference results.
    :param remove_prefixes: A set of string prefixes separated by comma list of string prefixes to be removed \
        from the inference results in sequence. This can also be used to remove the prompt from the inference \
        results. The prefixes should be separated by a comma.
    :param strip_characters: A set of characters to remove from the beginning or end of the extracted answer.\
        It is applied in the very end of the extraction process.
    :param extract_number: If the inference results contain a number, this can be used to extract the first or \
        last number in the inference results. The number will be extracted as a string.
    :param label_map: JSON serialized dictionary to perform mapping. Must contain key-value pair \
        "column_name": "<actual_column_name>" whose value needs mapping, followed by key-value pairs containing \
        idtolabel or labeltoid mappers. This is not aplicable to custom scripts.
    :param template: Jinja template containing the extraction logic of inference post-processing.
    :param script_path: Path to the custom preprocessor python script provided by user.
    :param output_dataset: Path to the jsonl file where the processed data will be saved.
    :return: None
    """
    processor = inferpp.BabelInferencePostProcessor(
        prediction_dir=prediction_dir,
        prediction_filename=prediction_filename if prediction_filename else "few_shot_prompt",
        prediction_dataset=prediction_dataset,
        prediction_column_name=prediction_column_name if prediction_column_name else 'prediction',
        separator=separator,
        find_first=find_first,
        regex_expr=regex_expr,
        remove_prefixes=remove_prefixes,
        strip_characters=strip_characters,
        extract_number=extract_number,
        label_map=label_map,
        remove_prompt_prefix=remove_prompt_prefix,
        output_dataset=os.path.join(output_dataset, OUTPUT_FILENAME),
        completion_key=completion_key
    )
    processor.run()
    log_mlflow_params(
        prediction_dataset=prediction_dataset,
        prediction_column_name=prediction_column_name,
        separator=separator if separator else None,
        find_first=find_first if find_first else None,
        regex_expr=regex_expr if regex_expr else None,
        remove_prefixes=remove_prefixes if remove_prefixes else None,
        strip_characters=strip_characters if strip_characters else None,
        extract_number=extract_number if extract_number else None,
        label_map=label_map if label_map else None,
        output_dataset=output_dataset,
    )
    return


if __name__ == "__main__":
    argss = parse_arguments()
    main(
        prediction_dir=argss.input_dir,
        prediction_dataset=argss.input_uri_file,
        prediction_filename=argss.input_filename,
        completion_key=argss.completion_key,
        prediction_column_name=argss.extracted_prediction_key,
        separator=argss.separator,
        find_first=argss.find_first,
        regex_expr=argss.regex,
        remove_prefixes=argss.remove_prefixes,
        remove_prompt_prefix=argss.remove_prompt_prefix,
        strip_characters=argss.strip_characters,
        extract_number=argss.extract_number,
        label_map=argss.label_map,
        output_dataset=argss.output_dataset,
    )
