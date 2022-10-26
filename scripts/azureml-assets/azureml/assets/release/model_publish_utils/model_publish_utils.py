import os
from pathlib import Path
import shutil
import stat
import logging
from datetime import date
import json
import requests
import logging
from typing import List
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

class ModelPublishUtils:

    @classmethod
    def __init__(self,model_id:str, model_dir: str):
        self.model_id = model_id
        self.model_dir = model_dir
    
    # Validate whether model name is valid with current task name
    @classmethod
    def validate_model_and_task(self) -> bool:
        response = self.__invoke_huggingface_api()
        isValid = True
        if response.get('status', None) == 'Failed':
            logging.error(response.message)
            isValid = False
        elif not self.__validateModelTaskName(response):
            logging.error("Invalid Task {} for the model {}".format(self.task_name, self.model_name))
            isValid = False
        return isValid

    @classmethod
    def __validateModelTaskName(self, response:dict) -> bool:
        if response.get('pipeline_tag', None) == self.task_name:
            return True
        response.keys()
        config_info = response.get('config', None)
        if config_info and type(config_info) == dict:
            task_specific_info = config_info.get('task_specific_params', None)
            return task_specific_info and type(task_specific_info) == dict and self.task_name in task_specific_info.keys()
        return False

    @classmethod
    def git_clone_model(self, model_uri: str, model_commit: str = None) -> None:
        if self.validate_model_and_task():
            cmd = f'git clone {model_uri} {self.model_dir}'
            os.system(cmd)
            if model_commit:
                os.chdir(f'{self.model_dir}')
                os.system(f'git reset --hard {model_commit}')

    #Get supported frameworks like pytorch, tensorflow
    @classmethod
    def get_supported_frameworks(self) -> List[str]:
        return [
            "pytorch",
            "tensorflow",
            "flax"
        ]

    # Convert Model to ISV FLavoured MLFlow model and save it.
    @classmethod
    def save_mlflow_model(self, model_dir_name: str):
        config = AutoConfig.from_pretrained(self.model_name)
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
            model = task_model_mapping[self.task_name].from_pretrained(self.model_name, config=config)
        elif "translation" in self.task_name:
            model = AutoModelWithLMHead.from_pretrained(self.model_name, config=config)
        else:
            logging.error("Invalid Task Name")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, config=config)
        sign_dict = {"inputs": '[{"name": "input_string", "type": "string"}]', "outputs": '[{"type": "string"}]'}
        if self.task_name == "question-answering":
            sign_dict["inputs"] = '[{"name": "question", "type": "string"}, {"name": "context", "type": "string"}]'
        signature = ModelSignature.from_dict(sign_dict)
        mlflow.hftransformers.save_model(model, f"{model_dir_name}", tokenizer, config, misc_conf, signature=signature)

    # Handle inference input based on input signature and task
    @classmethod
    def prepare_inference_input(self, input: str) -> str:
        maskToken = self.__getMaskTokenFor() if self.task_name == "fill-mask" else ""
        return f'{input}{maskToken}'

    @classmethod
    def __getMaskTokenFor(self) -> str:
        response = self.__invoke_huggingface_api()
        if response.get('status',None) == 'Failed':
            raise RuntimeError(f'Failed to fetch the mask token for the model {self.model_name}. Reason: {response.message}')
        return response.get('mask_token',"")

    @classmethod
    def __invoke_huggingface_api(self) -> dict:
        try:
            logging.info("Calling HuggingFace....")
            response = requests.get(f'https://huggingface.co/api/models/{self.model_name}')
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as exc:
            return {
                'status':'Failed',
                'message':str(exc.args)
            }
    
    # Get all libraries for incoming model
    @classmethod
    def get_dependencies_list(self) -> List[str]:
        return [
            "transformers[torch]"
        ]

    # Check whether model is updated
    @classmethod
    def isModelUpdated(self, registry_model) -> bool:
        #TODO
        pass

    # Get Feed url prefix for ISV
    @classmethod
    def get_registry_feedurl_prefix(self) -> str:
        return "HF"

    @classmethod
    def delete_handler(func, path, exc_info):
        print("We got the following exception")
        print(exc_info)

    @classmethod
    def delete_model_dir(self):
        shutil.rmtree(self.model_dir, ignore_errors=False, onerror=self.delete_handler)


        
