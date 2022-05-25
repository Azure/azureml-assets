import re
from enum import Enum
from pathlib import Path
from typing import Dict, List
from yaml import safe_load


class ValidationException(Exception):
    """ Validation errors """


TEMPLATE_CHECK = re.compile(r"\{\{.*\}\}")


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

    def _append_to_file_path(self, relative_path: Path) -> Path:
        return self.file_path / relative_path

    @staticmethod
    def _is_set(value: object):
        return value is not None

    @staticmethod
    def _contains_template(value: str):
        return TEMPLATE_CHECK.match(value) is not None

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
    extra_config: environment.yaml
    test:
      pytest:
        enabled: true
        pip_requirements: tests/requirements.txt
        tests_dir: tests
    """
    def __init__(self, file_name: Path):
        super().__init__(file_name)
        self._validate()

    def __str__(self) -> str:
        return f"{self.type.value} {self.name} {self.version}"

    def _validate(self):
        Config._validate_enum('type', self._type, AssetType, True)
        Config._validate_exists('spec', self.spec)
        Config._validate_exists('name', self.name)
        if not self.auto_version:
            Config._validate_exists('version', self.version)
        if self.type == AssetType.ENVIRONMENT:
            Config._validate_exists('extra_config', self.extra_config)

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
            name = Spec(self.spec_with_path).name
            if Config._contains_template(name):
                raise ValidationException(f"Tried to read asset name from spec, but it includes a template tag: {name}")
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
            version = Spec(self.spec_with_path).version
            if Config._contains_template(version):
                if self.auto_version:
                    version = None
                else:
                    raise ValidationException(f"Tried to read asset version from spec, but it includes a template tag: {version}")
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

    @property
    def extra_config(self) -> str:
        return self._yaml.get('extra_config')

    @property
    def extra_config_with_path(self) -> Path:
        config = self.extra_config
        return self._append_to_file_path(config) if config else None

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
      name: azureml/curated/tensorflow-2.7-ubuntu20.04-py38-cuda11-gpu # Can include registry hostname and template tags
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

        if self.context_dir and ".." in self.context_dir:
            raise ValidationException(f"Invalid context.dir property: {self.context_dir} refers to a parent directory")

    @property
    def _image(self) -> Dict[str, object]:
        return self._yaml.get('image', {})

    @property
    def image_name(self) -> str:
        return self._image.get('name')

    def get_image_name_with_tag(self, tag: str) -> str:
        return self._image.get('name') + f":{tag}"

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
        return [self._append_to_context_path(f) for f in self.template_files]

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
