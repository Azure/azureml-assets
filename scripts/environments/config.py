import os
from enum import Enum
from typing import Dict, List
from yaml import safe_load


class ValidationException(Exception):
    """ Validation errors """


class AssetType(Enum):
    CODE = 'code'
    COMPONENT = 'component'
    ENVIRONMENT = 'environment'
    MODEL = 'model'


class Config:
    def __init__(self, file_name: str):
        with open(file_name) as f:
            self._yaml = safe_load(f)
        self._file_name = file_name
        self._file_path = os.path.dirname(file_name)

    @property
    def file_path(self) -> str:
        return self._file_path

    def _append_to_file_path(self, relative_path: str) -> str:
        return os.path.join(self.file_path, relative_path)

    @staticmethod
    def _is_set(value: object):
        return value is not None

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


class AssetConfig(Config):
    """
    Example:

    type: environment
    definition: spec.yaml
    """
    def __init__(self, file_name: str):
        super().__init__(file_name)
        self._validate()

    def _validate(self):
        Config._validate_enum('type', self._type, AssetType, True)
        Config._validate_exists('spec', self.spec)

    @property
    def _type(self) -> str:
        return self._yaml.get('type')

    @property
    def type(self) -> AssetType:
        return AssetType(self._type)

    @property
    def spec(self) -> str:
        return self._yaml.get('spec')

    @property
    def spec_with_path(self) -> str:
        return self._append_to_file_path(self.spec)

    @property
    def extra_config(self) -> str:
        return self._yaml.get('extra_config')

    @property
    def extra_config_with_path(self) -> str:
        config = self.extra_config
        return self._append_to_file_path(config) if config else None


DEFAULT_CONTEXT_DIR = "context"
DEFAULT_DOCKERFILE = "Dockerfile"
DEFAULT_TEMPLATE_FILES = [DEFAULT_DOCKERFILE]
DEFAULT_ENVIRONMENT_VISIBLE = True


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


class EnvironmentConfig(Config):
    """
    Example:

    image:
      name: azureml/curated/tensorflow-2.7-ubuntu20.04-py38-cuda11-gpu
      os: linux
      context:
        dir: context
        dockerfile: Dockerfile
        pin_version_files:
        - Dockerfile
      publish:
        location: mcr
        visibility: public
    environment:
      visible: true
      metadata:
        os:
          name: Ubuntu
          version: "20.04"
    """
    def __init__(self, file_name: str):
        super().__init__(file_name)
        self._validate()

    def _validate(self):
        Config._validate_exists('image.name', self.image_name)
        Config._validate_enum('image.os', self._os, Os, True)

        if self._publish:
            Config._validate_enum('publish.location', self._publish_location, PublishLocation, True)
            Config._validate_enum('publish.visibility', self._publish_visibility, PublishVisibility, True)

    @property
    def _image(self) -> Dict[str, object]:
        return self._yaml.get('image', {})

    @property
    def image_name(self) -> str:
        return self._image.get('name')

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
    def context_dir(self) -> str:
        return self._context.get('dir', DEFAULT_CONTEXT_DIR)

    @property
    def context_dir_with_path(self) -> str:
        return self._append_to_file_path(self.context_dir)

    def _append_to_context_path(self, relative_path: str) -> str:
        return os.path.join(self.context_dir_with_path, relative_path)

    @property
    def dockerfile(self) -> str:
        return self._context.get('dockerfile', DEFAULT_DOCKERFILE)

    @property
    def dockerfile_with_path(self) -> str:
        return self._append_to_context_path(self.dockerfile)

    @property
    def template_files(self) -> List[str]:
        return self._context.get('template_files', DEFAULT_TEMPLATE_FILES)

    @property
    def template_files_with_path(self) -> List[str]:
        return [self._append_to_context_path(f) for f in self.template_files]

    @property
    def _publish(self) -> Dict[str, str]:
        return self._image.get('publish', {})

    @property
    def _publish_location(self) -> str:
        return self._publish.get('location')

    @property
    def publish_location(self) -> str:
        location = self._publish_location
        return PublishLocation(location) if location else None

    @property
    def _publish_visibility(self) -> str:
        return self._publish.get('visibility')

    @property
    def publish_visibility(self) -> str:
        visiblity = self._publish_visibility
        return PublishVisibility(visiblity) if visiblity else None

    @property
    def _environment(self) -> Dict[str, object]:
        return self._yaml.get('environment', {})

    @property
    def environment_visible(self) -> bool:
        return self._environment.get('visible', DEFAULT_ENVIRONMENT_VISIBLE)

    @property
    def environment_metadata(self) -> Dict[str, object]:
        return self._environment.get('metadata')
