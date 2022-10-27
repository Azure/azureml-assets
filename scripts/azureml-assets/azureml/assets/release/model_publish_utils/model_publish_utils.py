import os
import shutil
import yaml
import logging
from typing import List
from azureml.assets.util import logger
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
    def __init__(self, spec_path: str) -> None:
        
        self.model_yaml_path = spec_path + "/model.yml"
        self.asset_yaml_path = spec_path + "/asset.yml"
        self.model_name = self._get_model_name()
        self.model_uri = self._get_model_uri()
        self.model_commit = self._get_model_commit()
        self.model_task_name = self._get_model_task_name()
        self.model_dir_name = spec_path + self.model_name

    @classmethod
    def _get_model_name(self) -> str:
        with open(self.model_yaml_path) as f:
            model_file = yaml.safe_load(f)
        return model_file["name"]

    @classmethod
    def _get_model_uri(self) -> str:
        with open(self.asset_yaml_path) as f:
            asset_file = yaml.safe_load(f)
        return asset_file["package"]["path"]

    @classmethod
    def _get_model_commit(self) -> str/None:
        with open(self.asset_yaml_path) as f:
            asset_file = yaml.safe_load(f)
        return asset_file["package"]["sha"] if asset_file["package"]["sha"] else None
    
    @classmethod
    def _get_model_task_name(self) -> str:
        with open(self.asset_yaml_path) as f:
            asset_file = yaml.safe_load(f)
        return asset_file["task"]

    @classmethod
    def _git_clone_model(self) -> None:
        if self.validate_model_and_task():
            cmd = f'git clone {self.model_uri} {self.model_dir_name}'
            os.system(cmd)
            if self.model_commit:
                os.chdir(f'{self.model_dir_name}')
                os.system(f'git reset --hard {self.model_commit}')

    # Convert Model to ISV FLavoured MLFlow model and save it.
    @classmethod
    def _save_mlflow_model(self):
        config = AutoConfig.from_pretrained(self.model_name)
        misc_conf = {"task_type": self.model_task_name}
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
        if self.model_task_name in task_model_mapping:
            model = task_model_mapping[self.model_task_name].from_pretrained(self.model_name, config=config)
        elif "translation" in self.model_task_name:
            model = AutoModelWithLMHead.from_pretrained(self.model_name, config=config)
        else:
            logging.error("Invalid Task Name")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, config=config)
        sign_dict = {"inputs": '[{"name": "input_string", "type": "string"}]', "outputs": '[{"type": "string"}]'}
        if self.model_task_name == "question-answering":
            sign_dict["inputs"] = '[{"name": "question", "type": "string"}, {"name": "context", "type": "string"}]'
        signature = ModelSignature.from_dict(sign_dict)
        mlflow.hftransformers.save_model(model, f"{self.model_dir_name}", tokenizer, config, misc_conf, signature=signature)

    @classmethod
    def _change_model_path_in_yaml(self):
        with open(self.model_yaml_path) as f:
            model_file = yaml.safe_load(f)
        model_file['path'] = self.model_dir_name
        with open(self.model_yaml_path, 'w') as f:
            yaml.dump(model_file, f)

    @classmethod 
    def create_model_artifact(self):
        logger.print("Cloning the model artifacts into {self.model_dir_name}...")
        self._git_clone_model()
        logger.print("Converting custom model {self.model_name} into MLFlow Model...")
        self._save_mlflow_model()

    @classmethod
    def delete_model_artifact(self):
        logger.print("Deleting model files from disk")
        cmd = f'rm -rf {self.model_dir_name}'
        os.system(cmd)



        
