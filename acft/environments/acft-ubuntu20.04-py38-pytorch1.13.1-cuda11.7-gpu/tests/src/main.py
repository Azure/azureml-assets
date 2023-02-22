# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Sample Finetuning via ACFT."""
# imports
# flake8: noqa


import argparse
from azureml.acft.accelerator.finetune import AzuremlFinetuneArgs, AzuremlDatasetArgs, AzuremlTrainer
from azureml.acft.accelerator.constants import HfTrainerType
from datasets.load import load_dataset
from pathlib import Path
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import AutoConfig, AutoTokenizer # type: ignore
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score
)
from transformers.trainer_utils import EvalPrediction
from typing import Any, Dict


# <functions>
# define functions
def main(args):
    """Finetune model."""
    # read in data

    finetune_params = {
        "task_name": "SingleLabelClassification",
        "model_name": "roberta-base",
        "model_name_or_path": "roberta-base",
        "model_type": "roberta",
        "problem_type": "single_label_classification",
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "pad_to_max_length": True,
        "max_seq_length": 512,
        "ignore_mismatched_sizes": True,
        "pytorch_model_folder": "./outputs/acft_outputs/pytorch_model_folder",
        "output_dir": "./outputs/acft_outputs/pytorch_model_folder",
        "model_selector_output": "dummy_folder"
    }

    this_dir = Path(__file__).parent
    data_dir = Path.joinpath(this_dir, "data")
    train_file = str(Path.joinpath(data_dir, "train.jsonl"))
    valid_file = str(Path.joinpath(data_dir, "valid.jsonl"))
    dataset_dict = load_dataset("json", data_files={"train": train_file, "validation": valid_file})

    finetune_params["class_names"] = [0, 1]
    finetune_params["num_labels"] = len(finetune_params["class_names"])

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            finetune_params["model_name_or_path"],
            use_fast=True,
        )
    except Exception as e:
        print(f"Fast tokenizer not supported: {e}")
        print("Trying default tokenizer.")
        # slow tokenizer
        tokenizer =  AutoTokenizer.from_pretrained(
            finetune_params["model_name_or_path"],
        )
    print("Loaded tokenizer : {}".format(tokenizer))

    dataset_args = AzuremlDatasetArgs(
        train_dataset=dataset_dict["train"], # type: ignore
        validation_dataset=dataset_dict["validation"], # type: ignore
        data_collator=DataCollatorWithPadding(tokenizer) if tokenizer is not None else None
    )

    finetune_args = AzuremlFinetuneArgs(
        finetune_params,
        trainer_type=HfTrainerType.DEFAULT
    )

    class_names = finetune_params["class_names"]
    id2label = {idx: lbl for idx, lbl in enumerate(class_names)}
    label2id = {lbl: idx for idx, lbl in enumerate(class_names)}

    config = AutoConfig.from_pretrained(
        finetune_params["model_name_or_path"],
        problem_type=finetune_params["problem_type"],
        num_labels=finetune_params["num_labels"],
        id2label=id2label,
        label2id=label2id,
        output_attentions=False,
        output_hidden_states=False,
    )

    model, model_loading_metadata = AutoModelForSequenceClassification.from_pretrained(
        finetune_params["model_name_or_path"],
        config=config,
        ignore_mismatched_sizes=finetune_params.pop("ignore_mismatched_sizes", False),
        output_loading_info=True,
    )

    Path(finetune_params["output_dir"]).mkdir(exist_ok=True, parents=True)

    trainer = AzuremlTrainer(
        finetune_args=finetune_args,
        dataset_args=dataset_args,
        model=model,
        tokenizer=tokenizer,
        metric_func=single_label_metrics_func,
        new_initalized_layers=model_loading_metadata["missing_keys"],
    )

    trainer.train()


# </functions>


def single_label_metrics_func(eval_pred: EvalPrediction) -> Dict[str, Any]:
    """Compute and return metrics for sequence classification."""
    predictions, labels = eval_pred
    pred_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = labels.flatten()

    # NOTE `len` method is supported for torch tensor or numpy array. It is the count of elements in first dimension
    print(f"Predictions count: {len(pred_flat)} | References count: {len(labels_flat)}")
    accuracy = accuracy_score(y_true=labels_flat, y_pred=pred_flat)
    f1_macro = f1_score(y_true=labels_flat, y_pred=pred_flat, average="macro")
    mcc = matthews_corrcoef(y_true=labels_flat, y_pred=pred_flat)
    precision_macro = precision_score(y_true=labels_flat, y_pred=pred_flat, average="macro")
    recall_macro = recall_score(y_true=labels_flat, y_pred=pred_flat, average="macro")

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "mcc": mcc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro
    }

    return metrics


def parse_args():
    """Parse arguments."""
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--model-name", type=str, default="roberta-base", required=False)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
