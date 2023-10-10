from functools import partial
import json
import os
import logging
import numpy as np
import mlflow
import tqdm
from typing import Optional, List, Dict, Callable, Any

from .dataset_resolver import resolve_file
from .mlflow_logger import MLFlowLogger
from .extractor import (
    apply_separator,
    remove_prefix,
    remove_prefixes,
    apply_find_first,
    apply_extract_number,
    apply_replacements,
    extract_regex,
    apply_label_map,
    create_label_map,
    unpack_with_adjustment,
    batch_score_response_format_adjustment,
)

logger = logging.getLogger(__name__)
OUTPUT_FILENAME = "extracted_data.jsonl"

from typing import Optional


def get_completion_list_processor(
        data: Optional[Dict] = None,
        remove_prompt_prefix: Optional[bool] = False,
        prefixes: Optional[str] = None,
        separator: Optional[str] = None,
        keep_separator: Optional[bool] = False,
        find_first: Optional[str] = None,
        extract_number: Optional[str] = None,
        replacements: Optional[str] = None,
        regex: Optional[str] = None,
        lowercase: Optional[bool] = False,
        strip: Optional[bool] = False,
        label_map: Optional[Dict[str, Any]] = None) -> Callable:
    
    prompt_prefix = data["prompt"] if remove_prompt_prefix and "prompt" in data else None

    processors = [
        partial(remove_prefix, prefix=prompt_prefix) if prompt_prefix else None,
        partial(remove_prefixes, prefixes=prefixes) if prefixes else None,
        partial(apply_separator, separator=separator, keep_separator=keep_separator) if separator else None,
        partial(apply_find_first, candidates=find_first.split(",")) if find_first else None,
        partial(apply_extract_number, strategy=extract_number) if extract_number else None,
        partial(apply_replacements, replacements=replacements) if replacements else None,
        partial(extract_regex, regex=regex) if regex else None,
        (lambda x: x.lower()) if lowercase else None,
        (lambda x: x.strip(strip)) if strip else None,
        partial(apply_label_map, label_map=label_map) if label_map else None,
    ]
    def process_completion_list(completion_list: List[str]) -> List[str]:
        import pdb;pdb.set_trace();
        for processor in processors:
            if processor:
                completion_list = [processor(completion) for completion in completion_list]
        return completion_list
    
    return process_completion_list

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

    # Log the parameters first.
    mlflow.log_dict(locals(), "params.json")
    mlflow_logger = MLFlowLogger()

    # If input_uri_file is provided, use that
    if input_uri_file:
        input_path = input_uri_file
    else:
        input_path = resolve_file(input_path=input_dir, filename=input_filename)
    logger.info(f"Input path: {input_path}")

    if is_batch_score_pipeline:
        adjustment = partial(batch_score_response_format_adjustment, completion_key=completion_key)
    else:
        def do_nothing_adjustment(data):
            return data
        adjustment = do_nothing_adjustment

    # create output file
    output_path = os.path.join(output_dataset, OUTPUT_FILENAME)
    logger.info(f"Output path: {output_path}")

    # create label map
    label_map = create_label_map(label_map)
    logger.info(f"Label map: {label_map}")

    with open(output_path, "w") as f:
        with open(input_path) as input_f:

            for line in tqdm.tqdm(input_f):
                mlflow_logger.increment_step()

                data = unpack_with_adjustment(line, adjustment=adjustment)   # WHAT IS THIS ABOUT @Nivedita

                # It's possible for completions to be null e.g.
                # when doing streaming in a perf benchmark.
                completion_list = data.get(completion_key, None)

                # If the label key in the source data matches the completion key
                # from the inference component, it will be renamed to x_metadata
                # when the data is unpacked. We need to rename it back here, once
                # we have extracted the completion data.
                # Right now this is just for race_high_static and cnndm     -> LOOK AT THIS @Nivedita
                if f"{completion_key}_metadata" in data:
                    data[completion_key] = data[f"{completion_key}_metadata"]
                    del data[f"{completion_key}_metadata"]

                if isinstance(completion_list, str):
                    completion_list = [completion_list]
                if not completion_list:  # received no predictions
                    mlflow_logger.log_missing_completion()
                    continue

                raw_length = np.mean([len(str(completion)) for completion in completion_list])
                mlflow_logger.log_raw_completion_length(completion_length=raw_length)
                
                processor = get_completion_list_processor(
                    data=data,
                    remove_prompt_prefix=remove_prompt_prefix,
                    prefixes=prefixes,
                    separator=separator,
                    keep_separator=keep_separator,
                    find_first=find_first,
                    extract_number=extract_number,
                    replacements=replacements,
                    regex=regex,
                    lowercase=lowercase,
                    strip=strip,
                    label_map=label_map
                )
                processed_completion_list = processor(completion_list=completion_list)

                data[extracted_prediction_key] = processed_completion_list
                extracted_length = np.mean([len(str(completion)) for completion in processed_completion_list])  ## TO ADD
                mlflow_logger.log_extracted_completion_length(completion_length=extracted_length)
                data['raw_completion_length'] = raw_length
                data['extracted_completion_length'] = extracted_length

                f.write(json.dumps(data) + "\n")

            mlflow_logger.log_aggregates()
