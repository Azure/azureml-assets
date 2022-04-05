import os
from enum import Enum
from yaml import load, Loader

# Common
class AssetType(Enum):
    ENVIRONMENT = 'environment'

# Environment-specific
DEFAULT_CONTEXT_DIR = "context"
DEFAULT_DOCKERFILE = "Dockerfile"
DEFAULT_PIN_VERSION_FILES = [DEFAULT_DOCKERFILE]
OS_OPTIONS = ["linux", "windows"]


"""
Example:

type: environment
definition: my_env.yml
config:
  image_name: azureml/curated/tensorflow-2.7-ubuntu20.04-py38-cuda11-gpu
  os: linux
  context:
    dir: context
    dockerfile: Dockerfile
    pin_version_files:
    - Dockerfile
"""

class ValidationException(Exception):
    """ Validation errors """

class AssetConfig:
    def __init__(self, file_name: str):
        with open(file_name) as f:
            self._yaml = load(f, Loader=Loader)
        self._file_name = file_name
        self._file_path = os.path.dirname(file_name)
        self._validate()
    
    def _validate(self):
        if not self._raw_type:
            raise ValidationException("Missing 'type' property")
        asset_type_vals = [t.value for t in list(AssetType)]
        if self._raw_type not in asset_type_vals:
            raise ValidationException(f"Invalid 'type' property: {self._raw_type} is not in {asset_type_vals}")
        
        if not self.definition:
            raise ValidationException("Missing 'definition' property")
        
        if not self.config:
            raise ValidationException("Missing 'config' property")
    
    @property
    def file_name(self):
        return self._file_name

    @property
    def type(self):
        return AssetType(self._raw_type)

    @property
    def _raw_type(self):
        return self._yaml.get('type')

    @property
    def definition(self):
        return self._append_to_path(self._yaml.get('definition'))

    @property
    def config(self):
        return self._yaml.get('config')
    
    def _append_to_path(self, relative_path: str):
        return os.path.join(self._file_path, relative_path)

class EnvironmentConfig(AssetConfig):
    def __init__(self, asset_config: AssetConfig):
        self._yaml = asset_config._yaml
        self._validate()
    
    def _validate(self):
        if not self.config.image_name:
            raise ValidationException("Missing 'config.name' property")

        if not self.config.os:
            raise ValidationException("Missing 'config.os' property")
        elif self.config.os not in OS_OPTIONS:
            raise ValidationException(f"Invalid 'config.os' property: {self.config.os} is not in {OS_OPTIONS}")
    
    @property
    def image_name(self):
        return self.config.get('image_name')
    
    @property
    def os(self):
        return self.config.get('os')
    
    def _context(self):
        return self.config.get('context', {})

    @property
    def context_dir(self):
        return self._append_to_path(self._context().get('dir', DEFAULT_CONTEXT_DIR))
    
    def _append_to_context_path(self, relative_path: str):
        return os.path.join(self.context_dir, relative_path)

    @property
    def dockerfile(self):
        return self._append_to_context_path(self._context().get('dockerfile', DEFAULT_DOCKERFILE))
    
    @property
    def pin_version_files(self):
        relative_files = self._context().get('pin_version_files', DEFAULT_PIN_VERSION_FILES)
        return [self._append_to_context_path(f) for f in relative_files]