import os
import json
import pandas as pd
import torch
import azureml.evaluate.mlflow as mlflow
from transformers import pipeline


def init():

    global task_name, hf_pipeline

    model_path = str(os.getenv("AZUREML_MODEL_DIR"))
    model_path = os.path.join(model_path, "mlflow_model_folder")

    task_name, model, tokenizer, _ = mlflow.hftransformers.load_model(model_path)

    if task_name in ("multiclass", "multilabel"):
        hf_pipeline = pipeline(task="text-classification", model=model, tokenizer=tokenizer)
    else:
        hf_pipeline = pipeline(task=task_name, model=model, tokenizer=tokenizer)


def run(data):
    data = json.loads(data)
    if task_name == "question-answering":
        result = hf_pipeline(question=data["question"], context=data["context"])

    elif "translation" in task_name or \
        task_name in ("text-classification", "multiclass", "multilabel", "fill-mask", "ner", "summarization", "text-generation"):
        if type(data) != str:
            if "input" in data:
                data = data["input"]
            elif "text" in data:
                data = data["text"]

        result = hf_pipeline(data)

        if task_name == "ner":
            for ne in result:
                ne['score'] = float(ne['score'])

    else:
        result = "Task not supported"

    return result