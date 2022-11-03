# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import azureml.evaluate.mlflow as hf_mlflow
from mlflow.models import ModelSignature
from transformers import AutoTokenizer, AutoConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelWithLMHead,
)


class MLFlowModelUtils:
    def __init__(self, name, task_name, flavor, mlflow_model_dir):
        self.name = name
        self.task_name = task_name
        self.mlflow_model_dir = mlflow_model_dir
        self.flavor = flavor

    def _convert_to_mlflow_hftransformers(self):
        config = AutoConfig.from_pretrained(self.name)
        misc_conf = {"task_type": self.task_name}
        task_model_mapping = {
            "multiclass": AutoModelForSequenceClassification,
            "multilabel": AutoModelForSequenceClassification,
            "fill-mask": AutoModelForMaskedLM,
            "ner": AutoModelForTokenClassification,
            "question-answering": AutoModelForQuestionAnswering,
            "summarization": AutoModelWithLMHead,
            "text-generation": AutoModelWithLMHead,
            "text-classification": AutoModelForSequenceClassification,
        }
        if self.task_name in task_model_mapping:
            model = task_model_mapping[self.task_name].from_pretrained(
                self.name, config=config
            )
        elif "translation" in self.task_name:
            model = AutoModelWithLMHead.from_pretrained(self.name, config=config)
        else:
            logging.error("Invalid Task Name")
        tokenizer = AutoTokenizer.from_pretrained(self.name, config=config)
        sign_dict = {
            "inputs": '[{"name": "input_string", "type": "string"}]',
            "outputs": '[{"type": "string"}]',
        }
        if self.task_name == "question-answering":
            sign_dict[
                "inputs"
            ] = '[{"name": "question", "type": "string"}, {"name": "context", "type": "string"}]'
        signature = ModelSignature.from_dict(sign_dict)

        try:
            hf_mlflow.hftransformers.save_model(
                model,
                f"{self.mlflow_model_dir}",
                tokenizer,
                config,
                misc_conf,
                signature=signature,
            )
        except MemoryError:
            logging.error("Memory Error")

    def _convert_to_mlflow_package(self):
        return None

    def covert_into_mlflow_model(self):
        if self.flavor == "hftransformers":
            self._convert_to_mlflow_hftransformers()
        # TODO add support for pyfunc. Pyfunc requires custom env file.
        else:
            self._convert_to_mlflow_package()
