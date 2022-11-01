# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from enum import Enum
from pathlib import Path
from typing import Dict, List
from yaml import safe_load, dump
from subprocess import PIPE, run, STDOUT
from util import logger
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


class ValidationException(Exception):
    """ Validation errors """


TEMPLATE_CHECK = re.compile(r"\{\{.*\}\}")
EXCLUDE_PREFIX = "!"


class Config:
    def __init__(self, file_name: Path):
        with open(file_name) as f:
            self._yaml = safe_load(f)
        self._file_name_with_path = file_name
        self._file_name = file_name.name
        self._file_path = file_name.parent

    @property
    def file_name(self) -> str:
        return self._file_name

    @property
    def file_name_with_path(self) -> Path:
        return self._file_name_with_path

    @property
    def file_path(self) -> Path:
        return self._file_path

    @property
    def release_paths(self) -> List[Path]:
        return [self.file_name_with_path]

    def _append_to_file_path(self, relative_path: Path) -> Path:
        return self.file_path / relative_path

    @staticmethod
    def _is_set(value: object):
        return value is not None

    @staticmethod
    def _contains_template(value: str):
        return TEMPLATE_CHECK.search(value) is not None

    @staticmethod
    def _validate_exists(property_name: str, property_value: object):
        if not Config._is_set(property_value):
            raise ValidationException(f"Missing {property_name} property")

    @staticmethod
    def _validate_enum(property_name: str, property_value: object, enum: Enum, required=False):
        # Ensure property exists
        if required:
            Config._validate_exists(property_name, property_value)
        elif not Config._is_set(property_value):
            return

        # Validate value is expected
        enum_vals = [i.value for i in list(enum)]
        if property_value not in enum_vals:
            raise ValidationException(f"Invalid {property_name} property: {property_value}"
                                      f" is not one of {enum_vals}")

    @staticmethod
    def _expand_path(path: Path) -> List[Path]:
        if not path.exists():
            raise ValidationException(f"{path} not found")
        if path.is_dir():
            contents = list(path.rglob("*"))
            if contents:
                return contents
        return [path]


class Spec(Config):
    """
    Example:

    name: my-asset
    version: 1
    """
    def __init__(self, file_name: str):
        super().__init__(file_name)
        self._validate()

    def __str__(self) -> str:
        return f"{self.name} {self.version}"

    def _validate(self):
        Config._validate_exists('name', self.name)
        Config._validate_exists('version', self.version)

        if self.code_dir and not self.code_dir_with_path.exists():
            raise ValidationException(f"code directory {self.code_dir} not found")

    @property
    def name(self) -> str:
        return self._yaml.get('name')

    @property
    def version(self) -> str:
        version = self._yaml.get('version')
        return str(version) if version is not None else None

    @property
    def description(self) -> str:
        return self._yaml.get('description')

    @property
    def tags(self) -> Dict[str, str]:
        return self._yaml.get('tags')

    @property
    def image(self) -> str:
        return self._yaml.get('image')

    @property
    def code_dir(self) -> str:
        return self._yaml.get('code')

    @property
    def code_dir_with_path(self) -> Path:
        dir = self.code_dir
        return self._append_to_file_path(dir) if dir else None

    @property
    def release_paths(self) -> List[Path]:
        release_paths = super().release_paths
        code_dir = self.code_dir_with_path
        if code_dir:
            release_paths.extend(Config._expand_path(code_dir))
        return release_paths


class ModelConfig(Config):
    """

    Example:

        name: bert-base-uncased
        path: # should be string or should contain package details
            package: # if package not specificed , path would be be used like os path object.
                url: https://huggingface.co/bert-base-uncased
                commit_hash: 5546055f03398095e385d7dc625e636cc8910bf2
        publish:
            type: mlflow_model # could be one of (custom_model, mlflow_model, triton_model)
            flavors: hftransformers # flavors should be specificed only for mlflow_model
            tags: # tags published to the registry
                isv : huggingface
                task: fill-mask
    """

    @property
    def name(self) ->str: 
        return self._yaml.get("name")

    @property
    def path(self) -> object:
        return self._yaml.get("path")

    @property
    def package(self) -> Dict[str,str]:
        return self.path.get("package") if type(self.path()!="str") else None
        
    @property
    def commit_hash(self) -> str:
        return self.package.get("commit_hash") if self.package() !=None else None

    @property
    def url(self) -> str:
        return self.package.get("url") if self.package() !=None else None

    @property
    def publish(self) -> Dict[str,object]:
        return self._yaml.get("publish")

    @property
    def type(self) -> str:
        return self.publish.get("type")
    
    @property
    def flavors(self) -> str:
        return self.publish.get("flavors") if self.publish.get("flavors") else None

    @property
    def tags(self) -> Dict[str,str]:
        return self.publish("tags") if self.publish.get("tags") else {}
    
    @property
    def task_name(self) -> str:
        return self.tags.get("task") if self.tags.get("task") else None

    @property
    def model_dir(self) -> str:
        return "/tmp/" + self.name()
    
    def _download_model(self) -> None:

        cmd = f'git clone {self.url} {self.model_dir}'
        run(cmd)
        if self.commit_hash:
            run(f'cd {self.model_dir}')
            run(f'git reset --hard {self.commit_hash}')
    
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

    def _covert_into_mlflow_model(self):
        if self.flavors == "hftransformers":
            self._convert_to_mlflow_hftransformers()
        #TODO add support for pyfunc. Pyfunc requires custom env file.
        else :
            self._convert_to_mlflow_package()

    def prepare(self) -> str :
        """
        Prepares the model. Downloads the model if required and converts the models to specified
        publish type.

        Return: returns the local path to the model.
        """
        if (type(self.path) == str):
            return self.path

        self._download_model(self.url, self.commit_hash, self.model_dir)

        if self.type is 'mlflow_model':
            self._covert_into_mlflow_model(self.model_dir)        
        return self.model_dir

    def clean(self):
        """
            Deletes the Model Artifact after the model has been pushed to the registry
        """
        print("Deleting model files from disk")
        cmd = f'rm -rf {self.model_dir}'
        run(cmd)


DEFAULT_DOCKERFILE = "Dockerfile"
DEFAULT_TEMPLATE_FILES = [DEFAULT_DOCKERFILE]


class Os(Enum):
    LINUX = 'linux'
    WINDOWS = 'windows'


class PublishLocation(Enum):
    MCR = 'mcr'


class PublishVisibility(Enum):
    PUBLIC = 'public'
    INTERNAL = 'internal'
    STAGING = 'staging'
    UNLISTED = 'unlisted'


# Associates publish locations with their hostnames
PUBLISH_LOCATION_HOSTNAMES = {
    PublishLocation.MCR: 'mcr.microsoft.com'
}


class EnvironmentConfig(Config):
    """
    Example:

    image:
      name: azureml/curated/tensorflow-2.7-ubuntu20.04-py38-cuda11-gpu # Can include registry hostname & template tags
      os: linux
      context: # If not specified, image won't be built
        dir: context
        dockerfile: Dockerfile
        pin_version_files:
        - Dockerfile
      publish: # If not specified, image won't be published
        location: mcr
        visibility: public
    environment:
      metadata:
        os:
          name: Ubuntu
          version: "20.04"
    """
    def __init__(self, file_name: Path):
        super().__init__(file_name)
        self._validate()

    def _validate(self):
        Config._validate_exists('image.name', self.image_name)
        Config._validate_enum('image.os', self._os, Os, True)

        if self._publish:
            Config._validate_enum('publish.location', self._publish_location, PublishLocation, True)
            Config._validate_enum('publish.visibility', self._publish_visibility, PublishVisibility, True)

        if self.context_dir and not self.context_dir_with_path.exists():
            raise ValidationException(f"context.dir directory {self.context_dir} not found")

    @property
    def _image(self) -> Dict[str, object]:
        return self._yaml.get('image', {})

    @property
    def image_name(self) -> str:
        return self._image.get('name')

    def get_image_name_with_tag(self, tag: str) -> str:
        return f"{self.image_name}:{tag}"

    def get_full_image_name(self, default_tag: str = None) -> str:
        image = self.image_name

        # Only add tag if there's not already one in image name
        if default_tag and ":" not in image:
            image = f"{image}:{default_tag}"

        # Add hostname if publishing, otherwise it should already be in image
        hostname = self.publish_location_hostname
        if hostname:
            image = f"{hostname}/{image}"
        return image

    def get_image_name_for_promotion(self, tag: str = None) -> str:
        # Only promotion to MCR is supported
        if self.publish_location != PublishLocation.MCR:
            return None

        image = f"{self.publish_visibility.value}/{self.image_name}"
        if tag:
            image += f":{tag}"
        return image

    @property
    def _os(self) -> str:
        return self._image.get('os')

    @property
    def os(self) -> Os:
        return Os(self._os)

    @property
    def _context(self) -> Dict[str, object]:
        return self._image.get('context', {})

    @property
    def build_enabled(self) -> bool:
        return bool(self._context)

    @property
    def context_dir(self) -> str:
        return self._context.get('dir')

    @property
    def context_dir_with_path(self) -> Path:
        dir = self.context_dir
        return self._append_to_file_path(dir) if dir else None

    def _append_to_context_path(self, relative_path: Path) -> Path:
        dir = self.context_dir_with_path
        return dir / relative_path if dir else None

    @property
    def dockerfile(self) -> str:
        return self._context.get('dockerfile', DEFAULT_DOCKERFILE)

    @property
    def dockerfile_with_path(self) -> Path:
        return self._append_to_context_path(self.dockerfile)

    @property
    def template_files(self) -> List[str]:
        return self._context.get('template_files', DEFAULT_TEMPLATE_FILES)

    @property
    def template_files_with_path(self) -> List[Path]:
        files = [self._append_to_context_path(f) for f in self.template_files]
        return [f for f in files if f is not None]

    @property
    def release_paths(self) -> List[Path]:
        release_paths = super().release_paths
        context_dir = self.context_dir_with_path
        if context_dir:
            release_paths.extend(Config._expand_path(context_dir))
        return release_paths

    @property
    def _publish(self) -> Dict[str, str]:
        return self._image.get('publish', {})

    @property
    def _publish_location(self) -> str:
        return self._publish.get('location')

    @property
    def publish_location(self) -> PublishLocation:
        location = self._publish_location
        return PublishLocation(location) if location else None

    @property
    def publish_location_hostname(self) -> str:
        location = self._publish_location
        return PUBLISH_LOCATION_HOSTNAMES[PublishLocation(location)] if location else None

    @property
    def _publish_visibility(self) -> str:
        return self._publish.get('visibility')

    @property
    def publish_visibility(self) -> PublishVisibility:
        visiblity = self._publish_visibility
        return PublishVisibility(visiblity) if visiblity else None

    @property
    def _environment(self) -> Dict[str, object]:
        return self._yaml.get('environment', {})

    @property
    def environment_metadata(self) -> Dict[str, object]:
        return self._environment.get('metadata')


class AssetType(Enum):
    CODE = 'code'
    COMPONENT = 'component'
    ENVIRONMENT = 'environment'
    MODEL = 'model'


DEFAULT_ASSET_FILENAME = "asset.yaml"
VERSION_AUTO = "auto"


class AssetConfig(Config):
    """
    Example:

    name: my-asset
    version: 1 # Can also be set to auto to auto-increment version
    type: environment
    spec: spec.yaml
    extra_config: environment.yaml or model.yaml
    release_paths: # Additional dirs/files to include in release
    - ../src
    - !../src/test # Exclude by ! prefix
    test:
      pytest:
        enabled: true
        pip_requirements: tests/requirements.txt
        tests_dir: tests
    """
    def __init__(self, file_name: Path):
        super().__init__(file_name)
        self._spec = None
        self._extra_config = None
        self._validate()

    def __str__(self) -> str:
        return f"{self.type.value} {self.name} {self.version}"

    def _validate(self):
        Config._validate_enum('type', self._type, AssetType, True)
        Config._validate_exists('spec', self.spec)
        Config._validate_exists('name', self.name)
        if not self.auto_version:
            Config._validate_exists('version', self.version)
        if self.type == AssetType.ENVIRONMENT or self.type == AssetType.MODEL:
            Config._validate_exists('extra_config', self.extra_config)

        if not self.spec_with_path.exists():
            raise ValidationException(f"spec file {self.spec} not found")

        if self.extra_config and not self.extra_config_with_path.exists():
            raise ValidationException(f"extra_config file {self.extra_config} not found")

        include_paths = self._release_paths_includes_with_path
        if include_paths:
            missing = [p for p in include_paths if not p.exists()]
            if missing:
                raise ValidationException(f"missing release_paths: {missing}")

    @property
    def _type(self) -> str:
        return self._yaml.get('type')

    @property
    def type(self) -> AssetType:
        return AssetType(self._type)

    @property
    def _name(self) -> str:
        return self._yaml.get('name')

    @property
    def name(self) -> str:
        """Retrieve the asset's name from its YAML file, falling back to the spec if not set.

        Raises:
            ValidationException: If the name isn't set in the asset's YAML file and the name from spec includes a
                template tag.

        Returns:
            str: The asset's name
        """
        name = self._name
        if not Config._is_set(name):
            name = self.spec_as_object().name
            if Config._contains_template(name):
                raise ValidationException(f"Tried to read asset name from spec, "
                                          f"but it includes a template tag: {name}")
        return name

    @property
    def _version(self) -> str:
        return self._yaml.get('version')

    @property
    def version(self) -> str:
        """Retrieve the asset's version from its YAML file, falling back to the spec if not set.

        Raises:
            ValidationException: If the version isn't set in the asset's YAML file and the version from spec includes a
                template tag.

        Returns:
            str: The asset's version or None if auto-versioning and version from spec includes a template tag.
        """
        version = self._version
        if self.auto_version or not Config._is_set(version):
            version = self.spec_as_object().version
            if Config._contains_template(version):
                if self.auto_version:
                    version = None
                else:
                    raise ValidationException(f"Tried to read asset version from spec, "
                                              f"but it includes a template tag: {version}")
        return str(version) if version is not None else None

    @property
    def auto_version(self) -> bool:
        return self._version == VERSION_AUTO

    @property
    def spec(self) -> str:
        return self._yaml.get('spec')

    @property
    def spec_with_path(self) -> Path:
        return self._append_to_file_path(self.spec)

    def spec_as_object(self, force_reload: bool = False) -> Spec:
        if force_reload or self._spec is None:
            self._spec = Spec(self.spec_with_path)
        return self._spec

    @property
    def extra_config(self) -> str:
        return self._yaml.get('extra_config')

    @property
    def extra_config_with_path(self) -> Path:
        config = self.extra_config
        return self._append_to_file_path(config) if config else None

    def extra_config_as_object(self, force_reload: bool = False) -> Config:
        if force_reload or self._extra_config is None:
            if self.type == AssetType.ENVIRONMENT:
                self._extra_config = EnvironmentConfig(self.extra_config_with_path)
            elif self.type == AssetType.MODEL:
                self._extra_config = ModelConfig(self.extra_config_with_path)
        return self._extra_config

    def environment_config_as_object(self, force_reload: bool = False) :
        return self.extra_config_as_object(force_reload)

    @property
    def _release_paths(self) -> List[str]:
        return self._yaml.get('release_paths', [])

    @property
    def _release_paths_includes_with_path(self) -> Path:
        return [self._append_to_file_path(p) for p in self._release_paths if not p.startswith(EXCLUDE_PREFIX)]

    @property
    def _release_paths_excludes_with_path(self) -> Path:
        paths = [p[len(EXCLUDE_PREFIX):] for p in self._release_paths if p.startswith(EXCLUDE_PREFIX)]
        return [self._append_to_file_path(p) for p in paths]

    @property
    def release_paths(self) -> List[Path]:
        release_paths = super().release_paths

        # Collect files from spec
        release_paths.extend(self.spec_as_object().release_paths)

        # Collect files from extra_config if set
        extra_config = self.extra_config_as_object()
        if extra_config:
            release_paths.extend(extra_config.release_paths)

        # Expand release_paths
        for include_path in self._release_paths_includes_with_path:
            release_paths.extend(Config._expand_path(include_path))

        # Handle excludes
        exclude_paths = self._release_paths_excludes_with_path
        if exclude_paths:
            release_paths = [f for f in release_paths if not any(
                             [p for p in exclude_paths if p.exists() and (p.samefile(f) or p in f.parents)])]

        return release_paths

    @property
    def _test(self) -> Dict[str, object]:
        return self._yaml.get('test', {})

    @property
    def _test_pytest(self) -> Dict[str, object]:
        return self._test.get('pytest', {})

    @property
    def pytest_enabled(self) -> bool:
        return self._test_pytest.get('enabled', False)

    @property
    def pytest_pip_requirements(self) -> Path:
        return self._test_pytest.get('pip_requirements')

    @property
    def pytest_pip_requirements_with_path(self) -> Path:
        pip_requirements = self.pytest_pip_requirements
        return self._append_to_file_path(pip_requirements) if pip_requirements else None

    @property
    def pytest_tests_dir(self) -> Path:
        return self._test_pytest.get('tests_dir', ".") if self.pytest_enabled else None

    @property
    def pytest_tests_dir_with_path(self) -> Path:
        tests_dir = self.pytest_tests_dir
        return self._append_to_file_path(tests_dir) if tests_dir else None
