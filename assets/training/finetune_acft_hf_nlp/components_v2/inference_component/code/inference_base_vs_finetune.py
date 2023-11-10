import os
os.system("pip install openpyxl")

import argparse
from argparse import Namespace
from pathlib import Path

from datasets.load import load_dataset
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import time
import json
import pandas as pd
#from azureml.core import Run

#run = Run.get_context()

INPUT_COLUMN_KEY = "inputs"
PREDICTIONS_COLUMN_KEY = "predictions"
PREDICTIONS_FILE_NAME = "predictions.jsonl"
COMPARISON_FILE_NAME = "comparison_base_vs_fintuned.xlsx"
MODEL_DEVICE = "auto"
DATA_DEVICE = "cuda"


TOKENIZER_LOAD_KWARGS = {
    "clean_up_tokenization_spaces": True,
    # "padding": "max_length",
    # "padding_side": "right",
    # "truncation": True,
    # "return_full_text": False,
}
TOKENIZER_CALL_KWARGS = {
    # "padding": "max_length",
    # "truncation": True,
}
TOKENIZER_DECODE_KWARGS = {
    "skip_special_tokens": True,
    # "clean_up_tokenization_spaces": True,
}
GENERATION_CONFIG_KWARGS = {
    "do_sample": True
}


def save_predictions(predictions: dict, predictions_file_path: str) -> None:
    ds = Dataset.from_dict(predictions)
    ds.to_json(predictions_file_path)
    print(f"Predictions file path: {predictions_file_path}")


def get_predictions_for_text(model, tokenizer, text, input_generation_config):
    input_ids = tokenizer([text], return_tensors="pt", **TOKENIZER_CALL_KWARGS).input_ids.to(DATA_DEVICE)
    generation_config = None
    # if `generation_config` is not passed, getting overwritten
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and \
        getattr(model.base_model.model, "generation_config", None) is not None:
            generation_config = model.base_model.model.generation_config
    elif getattr(model, "generation_config", None) is not None:
        generation_config = model.generation_config

    if generation_config is None:
        generation_config=input_generation_config
    else:
        for key, value in input_generation_config.items():
            setattr(generation_config, key, value)

    outputs = model.generate(inputs=input_ids, **{"generation_config": generation_config})

    input_length = len(input_ids[0]) 
    total_length = len(outputs[0])

    # remove input_ids from outputs
    outputs = outputs[0].tolist()
    outputs = [outputs[len(input_ids[0]):]]

    predictions_text = tokenizer.batch_decode(
        outputs,
        **TOKENIZER_DECODE_KWARGS
    )
    return predictions_text, input_length, total_length


def generate_lora_predictions(args: Namespace, mock_request_body: dict, generation_config_list: list) -> dict:
    print(f"Base mlflow model path: {args.base_model_path}")
    tokenizer_path = str(Path(args.base_model_path, "data", "tokenizer"))
    model_path = str(Path(args.base_model_path, "data", "model"))

    print(f"Loading tokenizer from path: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path, **TOKENIZER_LOAD_KWARGS)
    print("Tokenizer loaded.")

    print(f"Loading model from path: {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=MODEL_DEVICE,
        **GENERATION_CONFIG_KWARGS,
    )
    print("Base model loaded.")

    model = base_model

    predictions = {
        PREDICTIONS_COLUMN_KEY: []
    }
    num_examples = len(mock_request_body[INPUT_COLUMN_KEY])
    generation_dict = {"Document": [], "generation_config": [], "BaseModelSummary": [], "FinetunedModelSummary": []}
    prev_time = time.time()
    for idx, text in enumerate(mock_request_body[INPUT_COLUMN_KEY]):
        for generation_config in generation_config_list:
            prediction, input_length, total_length = get_predictions_for_text(model, tokenizer, text, generation_config)
            predictions[PREDICTIONS_COLUMN_KEY].append(prediction[0])
            time_taken = time.time() - prev_time
            prev_time = time.time()
            print(f"\n{idx+1}/{num_examples}, input_length: {input_length}, total_length: {total_length}, {'{0:.2f}'.format(time_taken)}s/it")
            print(prediction)
            generation_dict["Document"].append(text)
            generation_dict["generation_config"].append(generation_config)
            generation_dict["BaseModelSummary"].append(prediction[0])

    if args.lora_model_path:
        print(f"Loading peft model from path: {args.lora_model_path}")
        model = PeftModel.from_pretrained(base_model, args.lora_model_path)
        print("PEFT model loaded.")
    prev_time = time.time()
    for idx, text in enumerate(mock_request_body[INPUT_COLUMN_KEY]):
        for generation_config in generation_config_list:
            prediction, input_length, total_length = get_predictions_for_text(model, tokenizer, text, generation_config)
            predictions[PREDICTIONS_COLUMN_KEY].append(prediction[0])
            time_taken = time.time() - prev_time
            prev_time = time.time()
            print(f"\n{idx+1}/{num_examples}, input_length: {input_length}, total_length: {total_length}, {'{0:.2f}'.format(time_taken)}s/it")
            print(prediction)
            generation_dict["FinetunedModelSummary"].append(prediction[0])

    return predictions, generation_dict


def get_mock_data_from_test_file(file_path: str, text_key: str, num_examples: int = 10) -> dict:
    mock_request = {
        INPUT_COLUMN_KEY: [],
    }
    # load jsonl data
    print(f"Loading data from path: {file_path}")
    ds = load_dataset("json", data_files=file_path, split="train")
    # convert dataset to request format
    ds_text = ds.map(lambda example: {INPUT_COLUMN_KEY: example[text_key]}, remove_columns=ds.column_names)[:num_examples]
    # convert to dict format
    mock_request.update(ds_text)  # ds_text.to_dict())
    # print(mock_request)
    print("Mock request prepared.")
    return mock_request

def generate_metrics(predictions, test_file_path, ground_truth_key):
    from azureml.metrics import compute_metrics, constants
    from pprint import pprint

    #y_pred = ["hello there general kenobi","foo bar foobar", "blue & red"]
    #y_test = [["hello there general kenobi san"], ["foo bar foobar"], ["blue & green"]]
    test_data = None

    if ground_truth_key:
        print(f"Loading data from path: {test_file_path}")
        ds = load_dataset("json", data_files=test_file_path, split="train")
        # convert dataset to request format
        ds_ground_truth = ds.map(lambda example: {ground_truth_key: example[ground_truth_key]}, remove_columns=ds.column_names)
        # convert to dict format
        test_data_dict = ds_ground_truth.to_dict()
        test_data = [[test] for test in test_data_dict[ground_truth_key]]

    try:
        result = compute_metrics(task_type=constants.Tasks.TEXT_GENERATION, y_test=test_data, y_pred=predictions)
        #run.log(result)
        pprint(result)
        return result
    except Exception:
        return


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--test_file_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--text_key",
        default="text",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ground_truth_key",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        required=False,
    )
    parser.add_argument(
        "--generation_config_list",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--output_dir",
        default="inference_output",
        type=str,
        help="folder to store inference output",
    )
    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    if args.generation_config_list:
        args.generation_config_list = json.loads(args.generation_config_list)
    else:
        args.generation_config_list = [{}]

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    args.predictions_file_path = str(Path(args.output_dir, PREDICTIONS_FILE_NAME))
    args.comparison_file_path = str(Path(args.output_dir, COMPARISON_FILE_NAME))

    mock_request_body = get_mock_data_from_test_file(args.test_file_path, args.text_key, args.num_examples)

    mock_response, generation_dict = generate_lora_predictions(args, mock_request_body, args.generation_config_list)

    metrics = generate_metrics(mock_response[PREDICTIONS_COLUMN_KEY], args.test_file_path, args.ground_truth_key)

    save_predictions(mock_response, args.predictions_file_path)
    pd.DataFrame(generation_dict).to_excel(args.comparison_file_path)


if __name__ == "__main__":
    main()
