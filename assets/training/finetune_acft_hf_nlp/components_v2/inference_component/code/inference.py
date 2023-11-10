import argparse
from argparse import Namespace
from pathlib import Path
from typing import Tuple, Dict

from datasets.load import load_dataset
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import time
import json


INPUT_COLUMN_KEY = "inputs"
PREDICTIONS_COLUMN_KEY = "predictions"
PREDICTIONS_FILE_NAME = "predictions.jsonl"
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
GENERATION_CONFIG_KWARGS = {}


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
            for key, value in input_generation_config.items():
                setattr(generation_config, key, value)
    outputs = model.generate(inputs=input_ids, **{"generation_config": input_generation_config})

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


def load_base_and_lora_model(args):
    model_path = str(Path(args.base_model_path, "data", "model"))
    print(f"Loading model from path: {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=MODEL_DEVICE,
        **GENERATION_CONFIG_KWARGS,
    )
    print("Base model loaded.")

    print(f"Loading peft model from path: {args.lora_model_path1}")
    peft_model1 = PeftModel.from_pretrained(base_model, args.lora_model_path1)
    print("PEFT model loaded.")

    print(f"Loading peft model from path: {args.lora_model_path2}")
    peft_model2 = PeftModel.from_pretrained(base_model, args.lora_model_path2)
    print("PEFT model loaded.")

    print(f"Loading peft model from path: {args.lora_model_path3}")
    peft_model3 = PeftModel.from_pretrained(base_model, args.lora_model_path3)
    print("PEFT model loaded.")

    return base_model, peft_model1, peft_model2, peft_model3


def generate_lora_predictions(args: Namespace, mock_request_body: dict, generation_config: dict, base_model, lora_model) -> dict:
    print(f"Base mlflow model path: {args.base_model_path}")
    tokenizer_path = str(Path(args.base_model_path, "data", "tokenizer"))
    # model_path = str(Path(args.base_model_path, "data", "model"))

    print(f"Loading tokenizer from path: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path, **TOKENIZER_LOAD_KWARGS)
    print("Tokenizer loaded.")

    # print(f"Loading model from path: {model_path}")
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     torch_dtype="auto",
    #     device_map=MODEL_DEVICE,
    #     **GENERATION_CONFIG_KWARGS,
    # )
    # print("Base model loaded.")

    num_examples = len(mock_request_body[INPUT_COLUMN_KEY])
    prev_time = time.time()
    for idx, text in enumerate(mock_request_body[INPUT_COLUMN_KEY]):
        prediction, input_length, total_length = get_predictions_for_text(base_model, tokenizer, text, generation_config)
        # predictions[PREDICTIONS_COLUMN_KEY].append(prediction[0])
        time_taken = time.time() - prev_time
        prev_time = time.time()
        print(f"\n{idx+1}/{num_examples}, input_length: {input_length}, total_length: {total_length}, {'{0:.2f}'.format(time_taken)}s/it")
        print(prediction)

    # print(f"Loading peft model from path: {args.lora_model_path}")
    # model = PeftModel.from_pretrained(base_model, args.lora_model_path)
    # print("PEFT model loaded.")

    predictions = {
        PREDICTIONS_COLUMN_KEY: []
    }
    num_examples = len(mock_request_body[INPUT_COLUMN_KEY])
    prev_time = time.time()
    for idx, text in enumerate(mock_request_body[INPUT_COLUMN_KEY]):
        prediction, input_length, total_length = get_predictions_for_text(peft_model, tokenizer, text, generation_config)
        predictions[PREDICTIONS_COLUMN_KEY].append(prediction[0])
        time_taken = time.time() - prev_time
        prev_time = time.time()
        print(f"\n{idx+1}/{num_examples}, input_length: {input_length}, total_length: {total_length}, {'{0:.2f}'.format(time_taken)}s/it")
        print(prediction)

    return predictions


def get_mock_data_from_test_file(file_path_or_dict: Tuple[str, Dict], text_key: str) -> dict:
    mock_request = {
        INPUT_COLUMN_KEY: [],
    }
    if isinstance(file_path_or_dict, str):
        # load jsonl data
        print(f"Loading data from path: {file_path_or_dict}")
        ds = load_dataset("json", data_files=file_path_or_dict, split="train")
        # convert dataset to request format
        ds_text = ds.map(lambda example: {INPUT_COLUMN_KEY: example[text_key]}, remove_columns=ds.column_names)
        # convert to dict format
        mock_request.update(ds_text.to_dict())
        # print(mock_request)
        print("Mock request prepared.")
    else:
        mock_request.update({INPUT_COLUMN_KEY: [file_path_or_dict[text_key]]})

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

    result = compute_metrics(task_type=constants.Tasks.TEXT_GENERATION, y_test=test_data, y_pred=predictions)
    pprint(result)
    return result


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--lora_model_path1",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--lora_model_path2",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--lora_model_path3",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--test_file_path",
        type=str,
        default=None,
        required=False,
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
        "--generation_config",
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
    base_model, lora_model1, lora_model2, lora_model3 = load_base_and_lora_model(args)

    if args.test_file_path:
        if args.generation_config:
            args.generation_config = json.loads(args.generation_config)
        else:
            args.generation_config = {}

        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        args.predictions_file_path = str(Path(args.output_dir, PREDICTIONS_FILE_NAME))

        mock_request_body = get_mock_data_from_test_file(args.test_file_path, args.text_key)

        mock_response = generate_lora_predictions(args, mock_request_body, args.generation_config, base_model, lora_model1)

        metrics = generate_metrics(mock_response[PREDICTIONS_COLUMN_KEY], args.test_file_path, args.ground_truth_key)

        save_predictions(mock_response, args.predictions_file_path)

        return

    while True:
        print("waiting for input!")
        inp_data = input()

        # convert inp_data to json
        inp_dict = json.loads(inp_data)
        generation_params = inp_dict.pop(
            "generation_params",
            {"do_sample": "true", "top_p": 0.9, "top_k": 50, "temperature": 0.9, "max_new_tokens": 256}
        )
        lora_model_id = map(int, inp_dict.pop("model_id", 1))
        lora_model = (
            lora_model1
            if lora_model_id == 1 else
            (
                lora_model2
                if lora_model_id == 2 else
                lora_model3
            )
        )

        mock_request_body = get_mock_data_from_test_file(inp_dict, args.text_key)
        mock_response = generate_lora_predictions(args, mock_request_body, generation_params, base_model, lora_model)


if __name__ == "__main__":
    main()
