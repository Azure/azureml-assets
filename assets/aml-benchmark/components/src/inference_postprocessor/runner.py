# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Entry script for 3P Inference Postprocessor Component."""

import argparse
import json
import os
import glob
from typing import Callable, Dict, Optional
from utils.logging import get_logger, log_mlflow_params
from utils.exceptions import swallow_all_exceptions

from utils.io import resolve_io_path, read_jsonl_files
from . import inference_postprocessor as inferpp

logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to dir to load the prediction dataset.",
    )
    parser.add_argument(
        "--input_filename",
        type=str,
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
        default=None,
        help="The key in the jsonl file that contains the inference results..",
    )
    parser.add_argument(
        "--extracted_prediction_key",
        type=str,
        required=True,
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
        "--prefixes",
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
        "--strip",
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
        "--output_dataset_result",
        type=str,
        default=None,
        help="Path to the jsonl file where the processed data will be saved.",
    )
    argss, _ = parser.parse_known_args()
    return argss


class BabelInferencePostProcessor(inferpp.InferencePostprocessor):
    def __init__(
        self,
        prediction_dataset: str = None,
        prediction_column_name: str = None,
        ground_truth_dataset: str = None,
        ground_truth_column_name: str = None,
        separator: str = None,
        find_first: str = None,
        regex_expr: str = None,
        remove_prefixes: str = None,
        strip_characters: str = None,
        label_map: str = None,
        template: str = None,
        user_postprocessor: str = None,
        output_dataset: str = None,
        extract_number: str = None,
        remove_prompt_prefix: str = False,
        prediction_dir: str = None,
        prediction_filename: str = "few_shot_prompt*",
        **kwargs
    ) ->None:
        super().__init__(
            prediction_dir=prediction_dir,
            prediction_filename=prediction_filename,
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
            output_dataset=output_dataset,
            **kwargs
        )

    def unpack_with_adjustment(line: str, adjustment: Callable[[Dict], Dict]) -> Dict:
        data = adjustment(json.loads(line))

        # flatten metadata
        if "metadata" in data:
            for k, v in data["metadata"].items():
                # Avoid accidental override of key in data
                key = f"{k}_metadata" if k in data else k
                data[key] = v
            del data["metadata"]
        
        if '_batch_request_metadata' in data:
            for k, v in data["_batch_request_metadata"].items():
                # Avoid accidental override of key in data
                key = f"{k}_metadata" if k in data else k
                data[key] = v
            del data["_batch_request_metadata"]
        
        return data

    def batch_score_response_format_adjustment(data, completion_key="samples"):
        """
        Because the response format is different between the scoring components,
        we need to adjust the schema for batch_score to be in line with other Babel components.
        """
        try:
            new_data = {
                "prompt": data["request"]["prompt"],
                completion_key: [sample["text"] for sample in data["response"]["choices"]],
            }
            if "request_metadata" in data:
                new_data["metadata"] = data["request_metadata"]
                if "completion" in new_data["metadata"]:
                    new_data["completion"] = new_data["metadata"]["completion"]
        except Exception:
            parsed_response = json.loads(data["response"])
            if "error" in parsed_response:
                logger.error(f"Error returned by the endpoint:\n{parsed_response['error']}")
            else:
                logger.exception("Something went wrong while converting schema.")
            new_data = data
        return new_data
    
    def resolve_file(input_path: str, filename: Optional[str] = None):
        """Resolve input path as single file from directory.

        Given input path can be either a file, or a directory. If its a file, it
        will be returned. If its a directory with a single file, that will be returned.
        If its a directory with multiple files and filename is provided, it will return
        the unique file matching the filename.

        Args:
            input_path (str): Either file or directory path
            filename (Optional[str]): If provided, will look for this file in dataset,
                assuming its a directory. Supports glob patterns.
        
        Examples:
            # my_dir contains only one file
            >>> resolve_file("my_dir")
            
            # my_dir contains multiple files
            >>> resolve_file("my_dir", "my_file.txt")
            
            # my_dir contains unique .txt file
            >>> resolve_file("my_dir", "*.txt")

        Returns:
            str: path to file
        """
        if os.path.isfile(input_path):
            logger.info(f"Found input file: {input_path}")
            return input_path

        if os.path.isdir(input_path):
            all_files = os.listdir(input_path)

            if not all_files:
                raise RuntimeError(f"Could not find any file in specified input directory {input_path}")

            if len(all_files) == 1:
                logger.info(f"Found input directory {input_path}, selecting unique file {all_files[0]}")
                return os.path.join(input_path, all_files[0])

            elif len(all_files) > 1 and filename is not None:

                logger.info(f"Found input directory {input_path}, selecting unique file {filename}")
                all_files = glob.glob(os.path.join(input_path, filename))
                if len(all_files) == 1:
                    return all_files[0]
                else:
                    raise RuntimeError(f"Found multiple files in input file path {input_path} for glob pattern {filename}")

            else:
                raise RuntimeError(f"Found multiple files in input file path {input_path}, specify the file name in addition.")

        logger.critical(f"Provided INPUT path {input_path} is neither a directory nor a file.")
        return input_path
    
    def format_metadata(self):
        if self.prediction_dataset: #self.input_uri_file:
            input_path = self.prediction_dataset
        else:
            input_path = self.resolve_file(input_path=self.prediction_dir, filename=self.prediction_filename)
        logger.info(f"Input path: {input_path}")

        dataset = 
        predicted_data = []
        with open(self.prediction_dataset, 'r') as input_f:


    def extract_inferences(self):
        """Extract inferences using generic method if no template or custom post-processor is provided."""
        predicted_data = read_jsonl_files(resolve_io_path(self.prediction_dataset))
        pred_list

        '''
        pred_list = []
        if self.prediction_column_name in predicted_data[0].keys():
            key = self.prediction_column_name
        else:
            key = key if key else "0"
        for row in predicted_data:
            predicted = row.get(key)
            if isinstance(predicted, list) and len(predicted[0]) > 1:
                curr_pred_list = []
                for i in range(0, len(predicted)):
                    out_string = predicted[i]
                    out_string = self.apply_remove_prompt_prefix(out_string, row)
                    out_string = self.apply_remove_prefixes(out_string)
                    out_string = self.apply_separator(out_string)
                    out_string = self.apply_find_first(out_string)
                    out_string = self.apply_extract_number(out_string)
                    out_string = self.apply_regex_expr(out_string)
                    out_string = self.apply_strip_characters(out_string)
                    out_string = self.apply_label_map(out_string)
                    curr_pred_list.append(out_string)
                pred_list.append(curr_pred_list)
            else:
                out_string = predicted if isinstance(predicted, str) else predicted[0]
                out_string = self.apply_remove_prompt_prefix(out_string, row)
                out_string = self.apply_remove_prefixes(out_string)
                out_string = self.apply_separator(out_string)
                out_string = self.apply_find_first(out_string)
                out_string = self.apply_extract_number(out_string)
                out_string = self.apply_regex_expr(out_string)
                out_string = self.apply_strip_characters(out_string)
                out_string = self.apply_label_map(out_string)
                pred_list.append(out_string)
        
        
        '''

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
        prediction_dir=prediction_dir,
        prediction_filename=prediction_filename,
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
        output_dataset=output_dataset,
        completion_key=completion_key
    )
    processor.run()
    # log_mlflow_params(  # Move this as Babel's design log the parameters first.
    #     prediction_dataset=prediction_dataset,
    #     prediction_column_name=prediction_column_name,
    #     ground_truth_dataset=ground_truth_dataset if ground_truth_dataset else None,
    #     ground_truth_column_name=ground_truth_column_name
    #     if ground_truth_column_name
    #     else None,
    #     separator=separator if separator else None,
    #     find_first=find_first if find_first else None,
    #     regex_expr=regex_expr if regex_expr else None,
    #     remove_prefixes=remove_prefixes if remove_prefixes else None,
    #     strip_characters=strip_characters if strip_characters else None,
    #     extract_number=extract_number if extract_number else None,
    #     label_map=label_map if label_map else None,
    #     template=template if template else None,
    #     user_postprocessor=script_path if script_path else None,
    #     output_dataset=output_dataset,
    # )
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
        remove_prefixes=argss.prefixes,
        remove_prompt_prefix=argss.remove_prompt_prefix,
        strip_characters=argss.strip,
        extract_number=argss.extract_number,
        label_map=argss.label_map
        output_dataset=argss.output_dataset_result,
    )

'''
def run_extractor(
    completion_key: str,
    extracted_prediction_key: str,
    output_dataset: str,
    input_dir: Optional[str] = None,
    input_filename: Optional[str] = None,
    input_uri_file: Optional[str] = None,
    separator: Optional[str] = None,
    keep_separator: Optional[bool] = False,
    prefixes: Optional[str] = None,
    remove_prompt_prefix: Optional[bool] = False,
    find_first: Optional[str] = None,
    extract_number: Optional[str] = None,
    replacements: Optional[str] = None,
    regex: Optional[str] = None,
    lowercase: Optional[bool] = False,
    strip: Optional[bool] = False,
    label_map: Optional[str] = None,
    is_batch_score_pipeline: Optional[bool] = False):


'''