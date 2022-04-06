import os
from enum import Enum
from yaml import load, Loader

# Common
class AssetType(Enum):
    COMPONENT = 'component'
    ENVIRONMENT = 'environment'
    MODEL = 'model'
    SNAPSHOT = 'snapshot'

# Environment-specific
DEFAULT_CONTEXT_DIR = "context"
DEFAULT_DOCKERFILE = "Dockerfile"
DEFAULT_TEMPLATE_FILES = [DEFAULT_DOCKERFILE]
OS_OPTIONS = ["linux", "windows"]
PUBLISH_VISIBILITIES = ["public", "internal", "staging", "unlisted"]


"""
Example:

type: environment
definition: spec.yaml
config:
  image_name: azureml/curated/tensorflow-2.7-ubuntu20.04-py38-cuda11-gpu
  os: linux
  context:
    dir: context
    dockerfile: Dockerfile
    pin_version_files:
    - Dockerfile
  publish:
    location: mcr
    visibility: public
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
        
        if not self.spec_with_path:
            raise ValidationException("Missing 'spec' property")
        
        if not self.config:
            raise ValidationException("Missing 'config' property")
    
    @property
    def file_name(self):
        return self._file_name

    @property
    def file_path(self):
        return self._file_path

    @property
    def type(self):
        return AssetType(self._raw_type)

    @property
    def _raw_type(self):
        return self._yaml.get('type')

    @property
    def spec_with_path(self):
        return self._append_to_path(self._yaml.get('spec'))

    @property
    def config(self):
        return self._yaml.get('config', {})
    
    def _append_to_path(self, relative_path: str):
        return os.path.join(self.file_path, relative_path)

class EnvironmentConfig(AssetConfig):
    def __init__(self, asset_config: AssetConfig):
        self._file_name = asset_config.file_name
        self._file_path = asset_config.file_path
        self._yaml = asset_config._yaml
        self._validate()
    
    def _validate(self):
        if not self.image_name:
            raise ValidationException("Missing 'config.image_name' property")

        if not self.os:
            raise ValidationException("Missing 'config.os' property")
        elif self.os not in OS_OPTIONS:
            raise ValidationException(f"Invalid 'config.os' property: {self.os} is not in {OS_OPTIONS}")
        
        if self.publish_location:
            if not self.publish_visibility:
                raise ValidationException("Missing 'config.publish.visibility' property")
            elif self.publish_visibility not in PUBLISH_VISIBILITIES:
                raise ValidationException(f"Invalid 'config.publish.visibility' property: {self.publish_visibility}"
                                          f" is not in {PUBLISH_VISIBILITIES}")
    
    @property
    def image_name(self):
        return self.config.get('image_name')
    
    @property
    def os(self):
        return self.config.get('os')
    
    @property
    def _context(self):
        return self.config.get('context', {})

    @property
    def context_dir_with_path(self):
        return self._append_to_path(self._context.get('dir', DEFAULT_CONTEXT_DIR))
    
    def _append_to_context_path(self, relative_path: str):
        return os.path.join(self.context_dir_with_path, relative_path)

    @property
    def dockerfile(self):
        return self._context.get('dockerfile', DEFAULT_DOCKERFILE)
    
    @property
    def dockerfile_with_path(self):
        return self._append_to_context_path(self.dockerfile)
    
    @property
    def template_files_with_path(self):
        relative_files = self._context.get('template_files', DEFAULT_TEMPLATE_FILES)
        return [self._append_to_context_path(f) for f in relative_files]
    
    @property
    def _publish(self):
        return self.config.get('publish', {})
    
    @property
    def publish_location(self):
        return self._publish.get('location')
    
    @property
    def publish_visibility(self):
        return self._publish.get('visibility')
    
    @property
    def metadata(self):
        return self.config.get('metadata')