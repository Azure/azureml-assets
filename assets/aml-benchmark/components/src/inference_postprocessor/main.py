# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Entry script for 3P Inference Postprocessor Component."""

import argparse

from utils.logging import get_logger, log_mlflow_params
from utils.exceptions import swallow_all_exceptions

from . import inference_postprocessor as inferpp

logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument(
        "--prediction_dataset",
        type=str,
        required=True,
        help="Path to load the prediction dataset.",
    )
    parser.add_argument(
        "--prediction_column_name",
        type=str,
        required=True,
        help="Prediction column name.",
    )
    parser.add_argument(
        "--ground_truth_dataset",
        type=str,
        default=None,
        help="Path to load the ground truth dataset.",
    )
    parser.add_argument(
        "--ground_truth_column_name",
        type=str,
        default=None,
        help="Ground truth column name.",
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
        "--regex_expr",
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
        "--template",
        type=str,
        default=None,
        help="Jinja template containing the extraction logic of inference post-processing.",
    )
    parser.add_argument(
        "--script_path",
        type=str,
        default=None,
        help="Path to the custom inference post-processor python script.",
    )
    parser.add_argument(
        "--output_dataset_result",
        type=str,
        default=None,
        help="Path to the jsonl file where the processed data will be saved.",
    )
    argss, _ = parser.parse_known_args()
    return argss


@swallow_all_exceptions(logger)
def main(
    prediction_dataset: str,
    prediction_column_name: str,
    ground_truth_dataset: str,
    ground_truth_column_name: str,
    separator: str,
    find_first: str,
    regex_expr: str,
    remove_prefixes: str,
    strip_characters: str,
    extract_number: str,
    label_map: str,
    template: str,
    script_path: str,
    output_dataset: str,
) -> None:
    """
    Entry function for Inference Postprocessor.

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
    processor = inferpp.InferencePostprocessor(
        prediction_dataset=prediction_dataset,
        prediction_column_name=prediction_column_name,
        ground_truth_dataset=ground_truth_dataset,
        ground_truth_column_name=ground_truth_column_name,
        separator=separator,
        find_first=find_first,
        regex_expr=regex_expr,
        remove_prefixes=remove_prefixes,
        strip_characters=strip_characters,
        extract_number=extract_number,
        label_map=label_map,
        template=template,
        user_postprocessor=script_path,
        output_dataset=output_dataset,
    )
    processor.run()
    log_mlflow_params(
        prediction_dataset=prediction_dataset,
        prediction_column_name=prediction_column_name,
        ground_truth_dataset=ground_truth_dataset if ground_truth_dataset else None,
        ground_truth_column_name=ground_truth_column_name
        if ground_truth_column_name
        else None,
        separator=separator if separator else None,
        find_first=find_first if find_first else None,
        regex_expr=regex_expr if regex_expr else None,
        remove_prefixes=remove_prefixes if remove_prefixes else None,
        strip_characters=strip_characters if strip_characters else None,
        extract_number=extract_number if extract_number else None,
        label_map=label_map if label_map else None,
        template=template if template else None,
        user_postprocessor=script_path if script_path else None,
        output_dataset=output_dataset,
    )
    return


if __name__ == "__main__":
    argss = parse_arguments()
    main(
        prediction_dataset=argss.prediction_dataset,
        prediction_column_name=argss.prediction_column_name,
        ground_truth_dataset=argss.ground_truth_dataset,
        ground_truth_column_name=argss.ground_truth_column_name,
        separator=argss.separator,
        find_first=argss.find_first,
        regex_expr=argss.regex_expr,
        remove_prefixes=argss.remove_prefixes,
        strip_characters=argss.strip_characters,
        extract_number=argss.extract_number,
        label_map=argss.label_map,
        template=argss.template,
        script_path=argss.script_path,
        output_dataset=argss.output_dataset_result,
    )
