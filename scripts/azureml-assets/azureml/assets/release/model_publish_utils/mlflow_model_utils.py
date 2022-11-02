# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import azureml.evaluate.mlflow as mlflow
from mlflow.models import ModelSignature
from sqlalchemy import true
from transformers import  AutoTokenizer, AutoConfig, pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelWithLMHead
)

class MLFlowModelUtils:

    def __init__(self, name, task_name, model_dir):
        self.name = name
        self.task_name = task_name
        self.model_dir = model_dir

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
            "text-classification": AutoModelForSequenceClassification
        }
        if self.task_name in task_model_mapping:
            model = task_model_mapping[self.task_name].from_pretrained(self.name, config=config)
        elif "translation" in self.task_name:
            model = AutoModelWithLMHead.from_pretrained(self.name, config=config)
        else:
            logging.error("Invalid Task Name")
        tokenizer = AutoTokenizer.from_pretrained(self.name, config=config)
        sign_dict = {"inputs": '[{"name": "input_string", "type": "string"}]', "outputs": '[{"type": "string"}]'}
        if self.task_name == "question-answering":
            sign_dict["inputs"] = '[{"name": "question", "type": "string"}, {"name": "context", "type": "string"}]'
        signature = ModelSignature.from_dict(sign_dict)
        self.mlflow_model_dir = self.model_dir + '/' + self.name + "-mlflow"
        mlflow.hftransformers.save_model(model, f"{self.mlflow_model_dir}", tokenizer, config, misc_conf, signature=signature) 

    def _convert_to_mlflow_package(self):
        return None       

    def covert_into_mlflow_model(self):
        if self.flavor == "hftransformers":
            self._convert_to_mlflow_hftransformers()
        #TODO add support for pyfunc. Pyfunc requires custom env file.
        else :
            self._convert_to_mlflow_package()