from yaml import load, Loader

DEFAULT_CONTEXT_DIR = "context"
DEFAULT_DOCKERFILE = "Dockerfile"
DEFAULT_PIN_VERSION_FILES = [DEFAULT_DOCKERFILE]
OS_OPTIONS = ["linux", "windows"]

"""
Example:

name: azureml/curated/tensorflow-2.7-ubuntu20.04-py38-cuda11-gpu
os: linux
context:
- dir: context
- dockerfile: Dockerfile
- pin_version_files:
  - Dockerfile
"""

class ValidationException(Exception):
    """ Validation errors """

class EnvConfig:
    def __init__(self, file_name: str):
        with open(file_name) as f:
            self._yaml = load(f, Loader=Loader)
        self._validate()
    
    def _validate(self):
        if not self.image_name:
            raise ValidationException("Missing 'name' property")

        if not self.os:
            raise ValidationException("Missing 'os' property")
        elif self.os not in OS_OPTIONS:
            raise ValidationException(f"Invalid 'os' property: {self.os} is not in {OS_OPTIONS}")
    
    @property
    def image_name(self):
        return self._yaml.get('name')
    
    @property
    def os(self):
        return self._yaml.get('os')
    
    def _context(self):
        return self._yaml.get('context', {})

    @property
    def context_dir(self):
        return self._context().get('dir', DEFAULT_CONTEXT_DIR)
    
    @property
    def dockerfile(self):
        return self._context().get('dockerfile', DEFAULT_DOCKERFILE)
    
    @property
    def pin_version_files(self):
        return self._context().get('pin_version_files', DEFAULT_PIN_VERSION_FILES)