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
        "--prediction_probabilities_dataset",
        type=str,
        default=None,
        help="Path to load the prediction probabilities dataset.",
    )
    parser.add_argument(
        "--encoder_config",
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
        "--separator",
        type=str,
        default=None,
        help="Few shot separator used in prompt crafter.",
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
        "--extract_value_at_index",
        type=int,
        default=None,
        help=(
            "If the regex_expr finds multiple strings matching the pattern in `regex_expr`, this"
            "can be used to extract the preferred value at a given index out of all matched patterns"
            "to be used as prediction value. If omitted, the default behaviour is first matched."
        ),
    )
    parser.add_argument(
        "--strip_prefix",
        type=str,
        default=None,
        help=(
            "Characters to remove from the beginning of the extracted answer."
            "It is applied in the very end of the extraction process."
        ),
    )
    parser.add_argument(
        "--strip_suffix",
        type=str,
        default=None,
        help=(
            "Characters to remove from the end of the extracted answer."
            "It is applied in the very end of the extraction process."
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
    prediction_probabilities_dataset: str,
    encoder_config: str,
    separator: str,
    regex_expr: str,
    extract_value_at_index: int,
    strip_prefix: str,
    strip_suffix: str,
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
    :param prediction_probabilities_dataset: Path to the jsonl file to load the prediction probabilities dataset.
    :param encoder_config: JSON serialized dictionary to perform mapping. Must contain key-value pair \
        "column_name": "<actual_column_name>" whose value needs mapping, followed by key-value pairs containing \
        idtolabel or labeltoid mappers. This is not aplicable to custom scripts.
    :param separator: Few shot separator used in prompt crafter.
    :param regex_expr: A regex pattern to extract the answer from the inference results.
    :param extract_value_at_index: The matched regex pattern value to be extracted in case
        multiple strings are found using the pattern provided in parameter `regex_expr`.
    :param strip_prefix: Characters to remove from the beginning of the extracted answer.
    :param strip_suffix: "Characters to remove from the end of the extracted answer."
    :param template: Jinja template containing the extraction logic of inference post-processing.
    :param script_path: Path to the custom preprocessor python script provided by user.
    :param output_dataset: Path to the jsonl file where the processed data will be saved.
    :return: None
    """
    processor = inferpp.InferencePostprocessor(
        prediction_dataset=prediction_dataset,
        Y=prediction_column_name,
        ground_truth_dataset=ground_truth_dataset,
        y=ground_truth_column_name,
        pred_probs_dataset=prediction_probabilities_dataset,
        encoder_config=encoder_config,
        separator=separator,
        regex_expr=regex_expr,
        extract_value_at_index=extract_value_at_index,
        strip_prefix=strip_prefix,
        strip_suffix=strip_suffix,
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
        prediction_probabilities_dataset=prediction_probabilities_dataset
        if prediction_probabilities_dataset
        else None,
        encoder_config=encoder_config if encoder_config else None,
        separator=separator if separator else None,
        regex_expr=regex_expr if regex_expr else None,
        strip_prefix=strip_prefix if strip_prefix else None,
        strip_suffix=strip_suffix if strip_suffix else None,
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
        prediction_probabilities_dataset=argss.prediction_probabilities_dataset,
        encoder_config=argss.encoder_config,
        separator=argss.separator,
        regex_expr=argss.regex_expr,
        extract_value_at_index=argss.extract_value_at_index,
        strip_prefix=argss.strip_prefix,
        strip_suffix=argss.strip_suffix,
        template=argss.template,
        script_path=argss.script_path,
        output_dataset=argss.output_dataset_result,
    )
