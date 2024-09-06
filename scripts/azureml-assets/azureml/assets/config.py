# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Asset config classes."""

import re
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from enum import Enum
from functools import total_ordering
from pathlib import Path
from ruamel.yaml import YAML
from packaging import version
from typing import Dict, List, Set, Tuple, Union
import requests
import sys
import urllib.parse
from azure.ai.ml._azure_environments import (
    AzureEnvironments,
    _get_default_cloud_name,
    _get_storage_endpoint_from_metadata
)
from azure.identity import AzureCliCredential
from azure.storage.blob import (
    BlobServiceClient,
    ContainerClient,
    ContainerSasPermissions,
    generate_container_sas
)


class ValidationException(Exception):
    """Validation errors."""


class AssetType(Enum):
    """Asset type."""

    COMPONENT = 'component'
    DATA = 'data'
    ENVIRONMENT = 'environment'
    EVALUATIONRESULT = 'evaluationresult'
    MODEL = 'model'
    PROMPT = 'prompt'


class ComponentType(Enum):
    """Enum for component types."""

    PIPELINE = 'pipeline'  # A pipeline component which allows multi-stage jobs.
    PARALLEL = 'parallel'  # A parallel component, aka PRSv2.
    COMMAND = 'command'  # A command component.
    AUTOML = 'automl'  # An AutoML component.
    SWEEP = 'sweep'  # A sweep component.


class DataAssetType(Enum):
    """Enum for data asset types."""

    URI_FILE = 'uri_file'  # A single file.
    URI_FOLDER = 'uri_folder'  # A folder containing files.


class ModelFlavor(Enum):
    """Enum for the Flavors accepted in ModelConfig."""

    HFTRANSFORMERS = 'hftransformers'
    PYTORCH = 'pytorch'


class ModelTaskName(Enum):
    """Enum for the Task names accepted in ModelConfig."""

    FILL_MASK = 'fill_mask'
    MULTICLASS = 'multiclass'
    MULTILABEL = 'multilabel'
    NER = 'ner'
    QUESTION_ANSWERING = 'question-answering'
    SUMMARIZATION = 'summarization'
    TEXT_GENERATION = 'text-generation'
    TEXT_CLASSIFICATION = 'text-classification'


class ModelType(Enum):
    """Enum for the Model Types accepted in ModelConfig."""

    MLFLOW = 'mlflow_model'
    CUSTOM = 'custom_model'
    TRITON = 'triton_model'


class GenericAssetType(Enum):
    """Enum for generic asset types."""

    PROMPT = 'prompt'
    EVALUATIONRESULT = 'evaluationresult'


class Os(Enum):
    """Operating system types."""

    LINUX = 'linux'
    WINDOWS = 'windows'


class PathType(Enum):
    """Enum for path types supported for model publishing."""

    LOCAL = "local"  # Path to model files present locally.
    GIT = "git"      # Model hosted on a public GIT repo and can be cloned by GIT LFS.
    FTP = "ftp"      # <UNSUPPORTED> Model files hosted on a FTP endpoint.
    HTTP = "http"    # <UNSUPPORTED> Model files hosted on a HTTP endpoint.
    AZUREBLOB = "azureblob"  # Model files hosted on an AZUREBLOB blobstore with public read access.


class PublishLocation(Enum):
    """Image publishing locations."""

    MCR = 'mcr'


class PublishVisibility(Enum):
    """Image publishing visibility types."""

    PUBLIC = 'public'
    INTERNAL = 'internal'
    STAGING = 'staging'
    UNLISTED = 'unlisted'


DEFAULT_ASSET_FILENAME = "asset.yaml"
DEFAULT_DESCRIPTION_FILE = "description.md"
DEFAULT_DOCKERFILE = "Dockerfile"
DEFAULT_TEMPLATE_FILES = [DEFAULT_DOCKERFILE]
EXCLUDE_PREFIX = "!"
FULL_ASSET_NAME_DELIMITER = "/"
FULL_ASSET_NAME_TEMPLATE = "{type}/{name}/{version}"
GENERIC_ASSET_TYPES = [AssetType.EVALUATIONRESULT, AssetType.PROMPT]
PARTIAL_ASSET_NAME_TEMPLATE = "{type}/{name}"
PUBLISH_LOCATION_HOSTNAMES = {PublishLocation.MCR: 'mcr.microsoft.com'}
STANDARD_ASSET_TYPES = [AssetType.COMPONENT, AssetType.DATA, AssetType.ENVIRONMENT, AssetType.MODEL]
TEMPLATE_CHECK = re.compile(r"\{\{.*\}\}")
VERSION_AUTO = "auto"


class Config:
    """Base class for asset configs."""

    def __init__(self, file_name: Path):
        """Create base config object.

        Args:
            file_name (Path): Location of config file.
        """
        with open(file_name) as f:
            self._yaml = YAML().load(f)
        self._file_name_with_path = file_name
        self._file_name = file_name.name
        self._file_path = file_name.parent

    @property
    def file_name(self) -> str:
        """Name of config file."""
        return self._file_name

    @property
    def file_name_with_path(self) -> Path:
        """Location of config file."""
        return self._file_name_with_path

    @property
    def file_path(self) -> Path:
        """Directory containing config file."""
        return self._file_path

    @property
    def release_paths(self) -> List[Path]:
        """Files that are required to create this asset."""
        return [self.file_name_with_path]

    def _append_to_file_path(self, relative_path: Path) -> Path:
        """Append a relative path to the directory containing the config file.

        Args:
            relative_path (Path): Path to append.

        Returns:
            Path: New path, relative to the directory containing the config file.
        """
        return self.file_path / relative_path

    @staticmethod
    def _is_set(value: object):
        """Determine whether an object is not None.

        Args:
            value (object): Object to test.

        Returns:
            _type_: True if not None, otherwise False.
        """
        return value is not None

    @staticmethod
    def _contains_template(value: str):
        """Determine whether a string contains any templates.

        Args:
            value (str): String to test.

        Returns:
            _type_: True if a template is found, otherwise False.
        """
        return TEMPLATE_CHECK.search(value) is not None

    @staticmethod
    def _validate_exists(property_name: str, property_value: object):
        """Ensure a property is set.

        Args:
            property_name (str): Property name, used only in exception message.
            property_value (object): Property value.

        Raises:
            ValidationException: If the property value isn't set.
        """
        if not Config._is_set(property_value):
            raise ValidationException(f"Missing {property_name} property")

    @staticmethod
    def _validate_enum(property_name: str, property_value: object, enum: Enum, required=False):
        """Ensure an enum value is set and is expected.

        Args:
            property_name (str): Property name, used only in exception message.
            property_value (object): Property value.
            enum (Enum): Enum object to test against.
            required (bool, optional): True if property_value must be set. Defaults to False.

        Raises:
            ValidationException: If required and not set, or if value is unexpected.
        """
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
        """Convert a path to a list of files it contains.

        Args:
            path (Path): File or directory to expand.

        Raises:
            ValidationException: If the path doesn't exist.

        Returns:
            List[Path]: If path is a file, just return it.
                        Otherwise, return the files contained by the directory.
        """
        if not path.exists():
            raise ValidationException(f"{path} not found")
        if path.is_dir():
            contents = [p for p in path.rglob("*") if p.is_file()]
            return contents
        return [path]


class Spec(Config):
    """Load and access spec file properties.

    Example:
        name: my-asset
        version: 1
    """

    def __init__(self, file_name: str):
        """Create spec object.

        Args:
            file_name (str): Spec file to load and validate.
        """
        super().__init__(file_name)
        self._validate()

    def __str__(self) -> str:
        """Asset name and version."""
        return f"{self.name} {self.version}"

    def _validate(self):
        """Validate spec contents.

        Only basic validation is performed. Conformance with an asset's JSON schema is not tested.

        Raises:
            ValidationException: If validation failed.
        """
        Config._validate_exists('name', self.name)
        Config._validate_exists('version', self.version)

        if self.code_dir and not self.code_dir_with_path.exists():
            raise ValidationException(f"code directory {self.code_dir} not found")
        if self._data_path:
            path = self.data_path_with_path
            if path.exists():
                if self.type == DataAssetType.URI_FILE.value and path.is_dir():
                    raise ValidationException(f"type is {self.type} but {self._data_path} is a directory")
                elif self.type == DataAssetType.URI_FOLDER.value and not path.is_dir():
                    raise ValidationException(f"type is {self.type} but {self._data_path} is a file")
            else:
                raise ValidationException(f"data path {self._data_path} not found")

    @property
    def name(self) -> str:
        """Asset name."""
        return self._yaml.get('name')

    @property
    def version(self) -> str:
        """Asset version."""
        version = self._yaml.get('version')
        return str(version) if version is not None else None

    @property
    def description(self) -> str:
        """Asset description."""
        return self._yaml.get('description')

    @property
    def tags(self) -> Dict[str, str]:
        """Asset tags."""
        return self._yaml.get('tags')

    @property
    def image(self) -> str:
        """Environment image."""
        return self._yaml.get('image')

    @property
    def type(self) -> str:
        """Type of a particular asset.

        For eg:
            `custom_model` or `mlflow_model` for a model asset.
            `command`, `pipeline` etc. for a component asset.
            `uri_file`, `uri_folder` for a data asset.
        """
        return self._yaml.get('type')

    @property
    def code_dir(self) -> str:
        """Component code directory."""
        if self.type == ComponentType.PARALLEL.value:
            task = self._yaml.get('task')
            return None if task is None else task.get('code')
        return self._yaml.get('code')

    @property
    def code_dir_with_path(self) -> Path:
        """Component code directory, relative to spec file's parent directory."""
        dir = self.code_dir
        return self._append_to_file_path(dir) if dir else None

    @property
    def _data_path(self) -> str:
        """Data asset path."""
        return self._yaml.get('path')

    @property
    def data_path_with_path(self) -> Path:
        """Data asset path, relative to spec file's parent directory."""
        data_path = self._data_path
        return self._append_to_file_path(data_path) if data_path else None

    @property
    def generic_asset_data_path(self) -> str:
        """Data path for a generic asset."""
        if self.type == GenericAssetType.PROMPT.value:
            return self._yaml.get('data_uri')
        elif self.type == GenericAssetType.EVALUATIONRESULT.value:
            return self._yaml.get('path')
        else:
            return None

    @property
    def generic_asset_data_path_with_path(self) -> Path:
        """Data path for a generic asset, relative to spec file's parent directory."""
        dir = self.generic_asset_data_path
        return self._append_to_file_path(dir) if dir else None

    @property
    def release_paths(self) -> List[Path]:
        """Files that are required to create this asset."""
        release_paths = super().release_paths

        # Add files from components
        code_dir = self.code_dir_with_path
        if code_dir:
            release_paths.extend(Config._expand_path(code_dir))

        # Add files from data assets
        data_path = self.data_path_with_path
        if data_path:
            release_paths.extend(Config._expand_path(data_path))

        # Add files from generic assets
        data_path = self.generic_asset_data_path_with_path
        if data_path:
            release_paths.extend(Config._expand_path(data_path))

        return release_paths

    @property
    def inference_config(self) -> Dict[str, Dict[str, Union[str, int]]]:
        """Inference config."""
        return self._yaml.get('inference_config')

    @property
    def os_type(self) -> str:
        """OS type."""
        return self._yaml.get('os_type')

    @property
    def dependencies(self) -> Dict[AssetType, Set]:
        """List of asset dependencies."""
        deps = defaultdict(set)
        if self.type in [ComponentType.COMMAND.value, ComponentType.PARALLEL.value]:
            # Find environment dependencies
            environment = None
            if self.type == ComponentType.COMMAND.value:
                environment = self._yaml.get('environment')
            else:
                environment = self._yaml.get('task', {}).get('environment')
            if isinstance(environment, str):
                # Skip inline environments
                deps[AssetType.ENVIRONMENT].add(environment)
        elif self.type == ComponentType.PIPELINE.value:
            # Find component dependencies
            for _, job in self._yaml.get('jobs', {}).items():
                if job.get('type') == ComponentType.COMMAND.value:
                    component = job.get('component')
                    if isinstance(component, str):
                        # Skip inline environments
                        deps[AssetType.COMPONENT].add(component)

        return deps

    @property
    def properties(self) -> Dict[str, str]:
        """Asset properties."""
        return self._yaml.get('properties', {})


class AssetPath:
    """Asset path."""

    def __init__(self, asset_type: str, uri: str):
        """Initialize asset path.

        :param asset_type: path type. Valid values are [local, git, ftp, http, azure]
        :type: str
        :param uri: a valid URI to local or remote resource
        :type: str
        """
        self._uri = uri
        self._type = asset_type

    @property
    def uri(self) -> str:
        """Asset URI."""
        return self._uri

    @property
    def type(self) -> str:
        """Asset type."""
        return self._type


class LocalAssetPath(AssetPath):
    """Local asset path."""

    def __init__(self, uri: str):
        """Create a Local path of asset.

        :param uri: Path to local model relative to asset.yaml
        :type uri: str
        """
        super().__init__(PathType.LOCAL, uri=uri)


class AzureBlobstoreAssetPath(AssetPath):
    """Azure Blobstore asset path."""

    AZURE_CLOUD_SUFFIX = "core.windows.net"

    CONTAINER_ANONYMOUS_ACCESS_CHECK_TIMEOUT = 15

    AZURE_CLI_PROCESS_LOGIN_TIMEOUT = 60

    def __init__(self, storage_name: str, container_name: str, container_path: str):
        """Create a Blobstore path.

        :param storage_name: Blob container storage name
        :type storage_name: str
        :param container_name: Blob container name
        :type container_name: str
        :param container_path: Relative path of assets in blob container
        :type container_path: str
        """
        self._storage_name = storage_name
        self._container_name = container_name
        self._container_path = container_path.lstrip("/").rstrip("/")
        self._token = None
        self._uri = None

        # AzureCloud, USGov, and China clouds should all pull from the same endpoint
        # associated with AzureCloud. If the cloud is not one of these, then the
        # endpoint will be dynamically acquired based on the currently configured
        # cloud.
        if _get_default_cloud_name() in [AzureEnvironments.ENV_DEFAULT,
                                         AzureEnvironments.ENV_US_GOVERNMENT,
                                         AzureEnvironments.ENV_CHINA]:
            cloud_suffix = AzureBlobstoreAssetPath.AZURE_CLOUD_SUFFIX
        else:
            cloud_suffix = _get_storage_endpoint_from_metadata()
        self._account_uri = f"https://{storage_name}.blob.{cloud_suffix}"

        # Its possible that the account URL may need additional tweaking to add a SAS
        # token if the account does not allow for anonymous access. However, for
        # performance reasons take a lazy approach to determining this and wait
        # until a client of this class actually attempts to reference the URI
        super().__init__(
            PathType.AZUREBLOB,
            None,
        )

    def get_uri(self, token_expiration: timedelta = timedelta(hours=1)) -> str:
        """Get Asset URI.

        :param token_expiration: Amount of time until token expiration
        :type token_expiration: timedelta
        """
        # There are 3 values token can be set to
        # None = no attempts have been made to create a SAS token
        # Empty string = container is public (no SAS token required)
        # Neither of the above = the SAS token

        # Build the simple, non-SAS URI for quick use later in the function.
        # Its possible that the account URI may need additional tweaking to add a SAS
        # token if the account does not allow for anonymous access.
        uri = f"{self._account_uri}/{self._container_name}/{self._container_path}"

        # If a SAS token has been explicitly set, then assume that the URI+token is valid.
        # Simply append the token and return. If no SAS token is set, then proceed to rest of function.
        if self.token is not None:
            if self.token:
                uri += "?" + self.token
            return uri

        # The first test is to see if we can simply list the contents of the container
        # using a very simple and quick HTTP request. In order for this to work,
        # container must support anonymous read access. If this succeeds, we can
        # return the URI "as-is".
        try:
            list_container_url = f"{self._account_uri}/{self._container_name}?restype=container&comp=list&maxresults=1"
            response = requests.get(
                list_container_url,
                timeout=AzureBlobstoreAssetPath.CONTAINER_ANONYMOUS_ACCESS_CHECK_TIMEOUT
            )

            if response.status_code >= 200 and response.status_code <= 299:
                self._token = ""
                return uri
        except Exception:
            # If we fail pass through to the next approach
            pass

        # Our second approach is to use the azure python SDK to view the properties
        # of the container. If the container allows for anonymous access then we can
        # return the URI "as-is".
        #
        # This approach is slower than the first approach, which is why we
        # tried the simple HTTP request approach first.
        #
        # It also requires Azure Credentials to be configured which may or may
        # not be present depending on the execution environment. If these credentials
        # do not exist then fail gracefully, return the URI "as-is", and hope for the best.
        try:
            blob_service_client = BlobServiceClient(
                account_url=self._account_uri,
                credential=AzureCliCredential(
                    process_timeout=AzureBlobstoreAssetPath.AZURE_CLI_PROCESS_LOGIN_TIMEOUT
                )
            )
            container_client = blob_service_client.get_container_client(container=self._container_name)

            # If the container allows for anonymous access then we can return the URI "as-is"
            if container_client.get_container_properties().public_access is not None:
                self._token = ""
                return uri

            # Our final approach is to generate a SAS token for the container and append
            # it to the URI
            start_time = datetime.now(timezone.utc)
            expiry_time = start_time + token_expiration

            key = blob_service_client.get_user_delegation_key(start_time, expiry_time)

            self._token = generate_container_sas(
                account_name=self._storage_name,
                container_name=self._container_name,
                user_delegation_key=key,
                permission=ContainerSasPermissions(read=True, list=True),
                expiry=expiry_time,
                start=start_time
            )

            uri += "?" + self._token
            return uri

        except Exception as e:
            # If we fail then simply return the URI "as-is" and hope for the best
            print(f"Failed to generate SAS token for storage {self._storage_name} "
                  f"and container {self._container_name}: {e}",
                  file=sys.stderr)
            self._token = ""
            return uri

    def get_container_client(self) -> ContainerClient:
        """Get container client.

        Returns:
            ContainerClient: Container client object for the asset.
        """
        # Get URL to container, while preserving any SAS token
        model_uri = urllib.parse.urlparse(self.uri)
        container_uri = urllib.parse.urlunparse(model_uri._replace(path="/" + self._container_name))
        container_client: ContainerClient = ContainerClient.from_container_url(container_url=container_uri)
        return container_client

    def get_files(self, strip_container_prefix: bool = True) -> List[dict]:
        """Get the list of files that belong to the asset.

        Args:
            strip_container_path (bool, optional): Whether to strip the container prefix from the file names.
                                                   Defaults to True.

        Returns:
            List[dict]: List of files and their sizes. Dicts have keys `name` and `size`.
        """
        container_client = self.get_container_client()
        container_prefix = self._container_path + "/"
        blobs = container_client.list_blobs(name_starts_with=container_prefix)

        # Remove prefix if desired
        starting_pos = len(container_prefix) if strip_container_prefix else 0
        blobs = [{'name': blob.name[starting_pos:], 'size': blob.size} for blob in blobs]
        return blobs

    def get_file_contents(self, name: str, encoding: str = "UTF-8") -> Union[str, bytes]:
        """Retrieve contents of a file from the asset.

        Args:
            name (str): File name, relative to the container path.
            encoding (str, optional): Encoding to use when reading the file. Defaults to "UTF-8".

        Returns:
            Union[str, bytes]: File contents, as str if encoding is provided, otherwise bytes.
        """
        container_client = self.get_container_client()
        container_prefix = self._container_path + "/"
        file_contents = container_client.download_blob(container_prefix + name, encoding=encoding).readall()
        return file_contents

    @property
    def uri(self) -> str:
        """Asset URI. Value is cached after first call."""
        if self._uri is None:
            self._uri = self.get_uri()
        return self._uri

    @property
    def storage_name(self) -> str:
        """Storage name."""
        return self._storage_name

    @property
    def container_name(self) -> str:
        """Container name."""
        return self._container_name

    @property
    def container_path(self) -> str:
        """Container path."""
        return self._container_path

    @property
    def token(self) -> str:
        """SAS token."""
        return self._token

    @token.setter
    def token(self, value: str):
        """Set SAS token. Resets cached `uri` value."""
        self._token = value
        self._uri = None


class GitAssetPath(AssetPath):
    """GIT asset path."""

    def __init__(self, branch: str, uri: str):
        """Create a GIT repo path.

        :param branch: Git branch to checkout from
        :type branch: str
        :param uri: git clonable url of repo
        :type uri: str
        """
        self._branch = branch
        super().__init__(PathType.GIT, uri)


class ModelConfig(Config):
    """Model Config class.

    Example:
        path: # should contain local_path or should contain package object
            ##Local path example
            type: local
            uri: "../models/bert-base-uncased" # the local path to the model

            ## GIT path example
            type: git
            uri: https://huggingface.co/bert-base-uncased
            branch: main

            ## Azure Blobstore example
            type: azureblob
            storage_name: my_storage
            container_name: my_container
            container_path: foo/bar
        publish:
            description: model_card.md
            type: mlflow_model
    """

    def __init__(self, file_name: Path):
        """Initialize object for the Model Properties extracted from extra_config model.yaml.

        Args:
            file_name (Path): Model config file to load and validate.
        """
        super().__init__(file_name)
        self._path = None
        self._description = ""
        self._validate()

    def _validate(self):
        """Validate the yaml file."""
        Config._validate_exists('model.path', self.path)
        Config._validate_enum('model.path.type', self.path.type.value, PathType, True)
        Config._validate_exists('model.publish', self._publish)
        Config._validate_exists('model.description', self._description_file_path)
        if self._description_file_path and not self._description_file_path.exists():
            raise ValidationException(f"description_file {self._description_file_path} not found")
        Config._validate_enum('model.type', self._type, ModelType, True)

    @property
    def path(self) -> AssetPath:
        """Model Path."""
        if self._path:
            return self._path
        path = self._yaml.get('path', {})
        if path and path.get('type'):
            path_type = path.get('type')
            if path_type == PathType.AZUREBLOB.value:
                self._path = AzureBlobstoreAssetPath(
                    storage_name=path['storage_name'],
                    container_name=path['container_name'],
                    container_path=path['container_path'],
                )
            elif path_type == PathType.GIT.value:
                self._path = GitAssetPath(branch=path['branch'], uri=path['uri'])
            elif path_type == PathType.LOCAL.value:
                self._path = LocalAssetPath(local_path=path['uri'])
            elif path_type == PathType.HTTP.value or path_type == PathType.FTP.value:
                raise NotImplementedError("Support for HTTP and FTP is being added.")
        else:
            raise Exception("path parameters are invalid")
        return self._path

    @property
    def _publish(self) -> Dict[str, object]:
        """Model publish properties."""
        return self._yaml.get('publish')

    @property
    def _description_file_path(self) -> Path:
        """Model description file path."""
        return self._file_path / self._publish.get('description')

    @property
    def description(self) -> str:
        """Model description."""
        if self._description_file_path and not self._description:
            with open(self._description_file_path) as f:
                self._description = f.read()
        return self._description

    @property
    def _type(self) -> str:
        """Model Type."""
        return self._publish.get('type')

    @property
    def type(self) -> ModelType:
        """Model Type Enum."""
        type = self._type
        return ModelType(type) if type else None


class EnvironmentConfig(Config):
    """Environment config.

    Example:
        image:
          # Image name can include registry hostname & template tags
          name: azureml/curated/tensorflow-2.7-ubuntu20.04-py38-cuda11-gpu
          os: linux
          context: # If not specified, image won't be built
            dir: context
            dockerfile: Dockerfile
            pin_version_files:
            - Dockerfile
          publish: # If not specified, image won't be published
            location: mcr
            visibility: public
    """

    def __init__(self, file_name: Path):
        """Create environment config object.

        Args:
            file_name (Path): Environment config file to load and validate.
        """
        super().__init__(file_name)
        self._validate()

    def _validate(self):
        """Validate environment config.

        Raises:
            ValidationException: If validation fails.
        """
        Config._validate_exists('image.name', self.image_name)
        Config._validate_enum('image.os', self._os, Os, True)

        if self._publish:
            Config._validate_enum('publish.location', self._publish_location, PublishLocation, True)
            Config._validate_enum('publish.visibility', self._publish_visibility, PublishVisibility, True)

        if self.context_dir and not self.context_dir_with_path.exists():
            raise ValidationException(f"context.dir directory {self.context_dir} not found")

    @property
    def _image(self) -> Dict[str, object]:
        """Raw 'image' value."""
        return self._yaml.get('image', {})

    @property
    def image_name(self) -> str:
        """Image name."""
        return self._image.get('name')

    def get_image_name_with_tag(self, tag: str) -> str:
        """Get image name with provided tag.

        Args:
            tag (str): Tag to append to image name.

        Returns:
            str: Image name with tag.
        """
        return f"{self.image_name}:{tag}"

    def get_full_image_name(self, default_tag: str = None) -> str:
        """Get fully qualified image name, including registry hostname.

        Args:
            default_tag (str, optional): Tag to add if there's not already one in the image name. Defaults to None.

        Returns:
            str: Fully qualified image name.
        """
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
        """Get image name used for promotion to publishing location.

        Args:
            tag (str, optional): Tag to append. Defaults to None.

        Returns:
            str: Image name for publishing.
        """
        # Only promotion to MCR is supported
        if self.publish_location != PublishLocation.MCR:
            return None

        image = f"{self.publish_visibility.value}/{self.image_name}"
        if tag:
            image += f":{tag}"
        return image

    def get_dockerfile_contents(self) -> str:
        """Dockerfile contents."""
        with open(self.dockerfile_with_path, "r") as f:
            return f.read()

    @property
    def _os(self) -> str:
        """Raw 'os' value."""
        return self._image.get('os')

    @property
    def os(self) -> Os:
        """Operating system."""
        return Os(self._os)

    @property
    def _context(self) -> Dict[str, object]:
        """Raw 'context' value."""
        return self._image.get('context', {})

    @property
    def build_enabled(self) -> bool:
        """Whether image should be built."""
        return bool(self._context)

    @property
    def context_dir(self) -> str:
        """Raw 'context.dir' value."""
        return self._context.get('dir')

    @property
    def context_dir_with_path(self) -> Path:
        """Context dir appended to environment config's parent directory."""
        dir = self.context_dir
        return self._append_to_file_path(dir) if dir else None

    def _append_to_context_path(self, relative_path: Path) -> Path:
        """Append path to build context directory.

        Args:
            relative_path (Path): Path to append.

        Returns:
            Path: New path.
        """
        dir = self.context_dir_with_path
        return dir / relative_path if dir else None

    @property
    def dockerfile(self) -> str:
        """Raw 'dockerfile' location.

        Defaults to Dockerfile if not specified in environment config.
        """
        return self._context.get('dockerfile', DEFAULT_DOCKERFILE)

    @property
    def dockerfile_with_path(self) -> Path:
        """Dockerfile path appended to build context directory."""
        return self._append_to_context_path(self.dockerfile)

    @property
    def template_files(self) -> List[str]:
        """Files containing templates that should be replaced prior to release."""
        return self._context.get('template_files', DEFAULT_TEMPLATE_FILES)

    @property
    def template_files_with_path(self) -> List[Path]:
        """Paths to files containing templates that should be replaced prior to release."""
        files = [self._append_to_context_path(f) for f in self.template_files]
        return [f for f in files if f is not None]

    @property
    def release_paths(self) -> List[Path]:
        """Files that are required to create this asset."""
        release_paths = super().release_paths
        context_dir = self.context_dir_with_path
        if context_dir:
            release_paths.extend(Config._expand_path(context_dir))
        return release_paths

    @property
    def _publish(self) -> Dict[str, str]:
        """Raw 'image.publish' value."""
        return self._image.get('publish', {})

    @property
    def publish_enabled(self) -> bool:
        """Whether image should be published."""
        return bool(self._publish)

    @property
    def _publish_location(self) -> str:
        """Raw 'image.location' value."""
        return self._publish.get('location')

    @property
    def publish_location(self) -> PublishLocation:
        """Image publishing location."""
        location = self._publish_location
        return PublishLocation(location) if location else None

    @property
    def publish_location_hostname(self) -> str:
        """Hostname of the registry to which an image will be published."""
        location = self._publish_location
        return PUBLISH_LOCATION_HOSTNAMES[PublishLocation(location)] if location else None

    @property
    def _publish_visibility(self) -> str:
        """Raw 'publish.visibility' value."""
        return self._publish.get('visibility')

    @property
    def publish_visibility(self) -> PublishVisibility:
        """Image's publishing visibility type."""
        visibility = self._publish_visibility
        return PublishVisibility(visibility) if visibility else None


class DataConfig(Config):
    """Data Asset Config class.

    Example:
        # Remote storage path for asset data
        path:
            type: azureblob
            storage_name: my_storage
            container_name: my_container
            container_path: foo/bar
    """

    def __init__(self, file_name: Path):
        """Initialize object for the Data Asset Properties extracted from storage.yaml.

        Args:
            file_name (Path): Storage config file to load and validate.
        """
        super().__init__(file_name)
        self._path = None
        self._validate()

    def _validate(self):
        """Validate the yaml file."""
        Config._validate_exists('data_asset.path', self.path)
        Config._validate_enum('data_asset.path.type', self.path.type.value, PathType, True)

    @property
    def path(self) -> AssetPath:
        """Remote Storage Path (Azure Blob is the only supported type)."""
        if self._path:
            return self._path
        path = self._yaml.get('path', {})
        if path and path.get('type'):
            path_type = path.get('type')
            if path_type == PathType.AZUREBLOB.value:
                self._path = AzureBlobstoreAssetPath(
                    storage_name=path['storage_name'],
                    container_name=path['container_name'],
                    container_path=path['container_path'],
                )
            else:
                raise NotImplementedError("Unrecognized path type.")
        else:
            return None
        return self._path


class GenericAssetConfig(Config):
    """Generic Asset Config class.

    Example:
        # Remote storage path for asset data
        path:
            type: azureblob
            storage_name: my_storage
            container_name: my_container
            container_path: foo/bar
    """

    def __init__(self, file_name: Path):
        """Initialize object for the Generic Asset Properties extracted from storage.yaml.

        Args:
            file_name (Path): Storage config file to load and validate.
        """
        super().__init__(file_name)
        self._path = None
        self._validate()

    def _validate(self):
        """Validate the yaml file."""
        Config._validate_exists('generic_asset.path', self.path)
        Config._validate_enum('generic_asset.path.type', self.path.type.value, PathType, True)

    @property
    def path(self) -> AssetPath:
        """Remote Storage Path (Azure Blob is the only supported type)."""
        if self._path:
            return self._path
        path = self._yaml.get('path', {})
        if path and path.get('type'):
            path_type = path.get('type')
            if path_type == PathType.AZUREBLOB.value:
                self._path = AzureBlobstoreAssetPath(
                    storage_name=path['storage_name'],
                    container_name=path['container_name'],
                    container_path=path['container_path'],
                )
            else:
                raise NotImplementedError("Unrecognized path type.")
        else:
            return None
        return self._path


@total_ordering
class AssetConfig(Config):
    """Asset config file.

    Example:
        name: my-asset
        version: 1 # Can also be set to auto to auto-increment version
        type: environment
        spec: spec.yaml
        description_file: description.md # Path to optional description file
        extra_config: environment.yaml
        release_paths: # Additional dirs/files to include in release
        - ../src
        - !../src/test # Exclude by ! prefix
        test:
          pytest:
            enabled: true
            conda_environment: tests/conda.yaml # Optional, must install pytest if specified
            pip_requirements: tests/requirements.txt # Optional, additional packages required by tests
            tests_dir: tests
        categories: ["PyTorch", "Training"] # List of categories
    """

    def __init__(self, file_name: Path):
        """Create asset config object.

        Args:
            file_name (Path): File to load and validate.
        """
        super().__init__(file_name)
        self._spec = None
        self._extra_config = None
        self._validate()

    def __str__(self) -> str:
        """Asset type, name, and version."""
        return f"{self.type.value} {self.name} {self.version}"

    def __eq__(self, other) -> bool:
        """Determine whether two AssetConfig objects are equal."""
        if not isinstance(other, AssetConfig):
            return NotImplemented

        return (self.type.value, self.name, self.version) == (other.type.value, other.name, other.version)

    def __lt__(self, other) -> bool:
        """Determine whether an AssetConfig objects is less than another."""
        if not isinstance(other, AssetConfig):
            return NotImplemented

        # Compare the easy ones first
        if self.type.value != other.type.value:
            return self.type.value < other.type.value
        if self.name != other.name:
            return self.name < other.name

        # Reject auto-versioned assets
        if self.version is None or other.version is None:
            raise ValueError("Cannot compare auto-versioned assets")

        # Compare versions using packaging's version object
        return version.parse(self.version) < version.parse(other.version)

    def __hash__(self) -> int:
        """Hash an AssetConfig object."""
        return hash((self.type.value, self.name, self.version))

    def _validate(self):
        """Validate asset config.

        Raises:
            ValidationException: If validation fails.
        """
        Config._validate_enum('type', self._type, AssetType, True)
        Config._validate_exists('spec', self.spec)
        Config._validate_exists('name', self.name)
        if not self.auto_version:
            Config._validate_exists('version', self.version)
        if self.type == AssetType.ENVIRONMENT:
            Config._validate_exists('extra_config', self.extra_config)

        if not self.spec_with_path.exists():
            raise ValidationException(f"spec file {self.spec} not found")

        if self.description_file and not self.description_file_with_path.exists():
            raise ValidationException(f"description_file {self.description_file} not found")

        if self.extra_config and not self.extra_config_with_path.exists():
            raise ValidationException(f"extra_config file {self.extra_config} not found")

        include_paths = self._release_paths_includes_with_path
        if include_paths:
            missing = [p for p in include_paths if not p.exists()]
            if missing:
                raise ValidationException(f"missing release_paths: {missing}")

        if self.pytest_enabled:
            if self.pytest_conda_environment and self.pytest_pip_requirements:
                raise ValidationException(
                    "pytest.conda_environment and pytest.pip_requirements are mutually exclusive")
            if not self.pytest_tests_dir:
                raise ValidationException("pytest.tests_dir is required")

    @property
    def _type(self) -> str:
        """Raw 'type' value."""
        return self._yaml.get('type')

    @property
    def type(self) -> AssetType:
        """Asset type."""
        return AssetType(self._type)

    @property
    def _name(self) -> str:
        """Raw 'name' value."""
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
    def partial_name(self) -> str:
        """Asset name, including type."""
        return PARTIAL_ASSET_NAME_TEMPLATE.format(type=self.type.value, name=self.name)

    @property
    def full_name(self) -> str:
        """Full asset name, including type and version."""
        return FULL_ASSET_NAME_TEMPLATE.format(type=self.type.value, name=self.name, version=self.version)

    @staticmethod
    def parse_full_name(full_name: str) -> Tuple[AssetType, str, str]:
        """Parse a full name into its asset type, name, and version.

        Args:
            full_name (str): Full name to parse

        Returns:
            Tuple[assets.AssetType, str, str]: Asset type, name, and version
        """
        tag_parts = full_name.split(FULL_ASSET_NAME_DELIMITER)
        if len(tag_parts) != 3:
            raise ValueError(f"Invalid full name: {full_name}")

        return AssetType(tag_parts[0]), tag_parts[1], tag_parts[2]

    @property
    def _version(self) -> str:
        """Raw 'version' value."""
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
        """Whether auto versioning is enabled."""
        return self._version == VERSION_AUTO

    @property
    def spec(self) -> str:
        """Raw 'spec' value."""
        return self._yaml.get('spec')

    @property
    def spec_with_path(self) -> Path:
        """Asset's spec file."""
        return self._append_to_file_path(self.spec)

    @property
    def categories(self) -> List[str]:
        """List of categories."""
        return self._yaml.get('categories', [])

    def spec_as_object(self, force_reload: bool = False) -> Spec:
        """Retrieve asset's spec file as an object.

        Args:
            force_reload (bool, optional): If cached, reload the spec file. Defaults to False.

        Returns:
            Spec: Asset's spec object.
        """
        if force_reload or self._spec is None:
            self._spec = Spec(self.spec_with_path)
        return self._spec

    @property
    def description_file(self) -> str:
        """Raw 'description_file' value."""
        return self._yaml.get('description_file')

    @property
    def description_file_with_path(self) -> Path:
        """Asset's description file."""
        description_file = self.description_file
        if description_file is None:
            # Check for default file
            description_file_path = self._append_to_file_path(DEFAULT_DESCRIPTION_FILE)
            if not description_file_path.exists():
                description_file_path = None
        else:
            # Use specified file
            description_file_path = self._append_to_file_path(description_file)
        return description_file_path

    @property
    def extra_config(self) -> str:
        """Raw 'extra_config' value."""
        return self._yaml.get('extra_config')

    @property
    def extra_config_with_path(self) -> Path:
        """Extra config file appended to asset config file's parent directory."""
        config = self.extra_config
        return self._append_to_file_path(config) if config else None

    def extra_config_as_object(self, force_reload: bool = False) -> Config:
        """Retrieve extra config file as an object.

        Args:
            force_reload (bool, optional): If cached, reload the extra config file. Defaults to False.

        Raises:
            Exception: If loading an extra_config for the asset type is unimplemented.

        Returns:
            Config: Extra config object.
        """
        if force_reload or self._extra_config is None:
            extra_config_with_path = self.extra_config_with_path
            if extra_config_with_path:
                if self.type == AssetType.ENVIRONMENT:
                    self._extra_config = EnvironmentConfig(extra_config_with_path)
                elif self.type == AssetType.MODEL:
                    self._extra_config = ModelConfig(extra_config_with_path)
                elif self.type == AssetType.DATA:
                    self._extra_config = DataConfig(extra_config_with_path)
                elif self.type == AssetType.PROMPT:
                    self._extra_config = GenericAssetConfig(extra_config_with_path)
                else:
                    raise Exception(f"extra_config loading for asset type {self.type.value} is unimplemented")
        return self._extra_config

    @property
    def _release_paths(self) -> List[str]:
        """Raw 'release_paths' value."""
        return self._yaml.get('release_paths', [])

    @property
    def _release_paths_includes_with_path(self) -> Path:
        """Files that are required to create this asset, excluding those that start with !."""
        return [self._append_to_file_path(p) for p in self._release_paths if not p.startswith(EXCLUDE_PREFIX)]

    @property
    def _release_paths_excludes_with_path(self) -> Path:
        """Files that are required to create this asset, filtered to those that start with !."""
        paths = [p[len(EXCLUDE_PREFIX):] for p in self._release_paths if p.startswith(EXCLUDE_PREFIX)]
        return [self._append_to_file_path(p) for p in paths]

    @property
    def release_paths(self) -> List[Path]:
        """Files that are required to create this asset."""
        release_paths = super().release_paths

        # Collect files from spec
        release_paths.extend(self.spec_as_object().release_paths)

        # Collect description file if set or found
        description_file = self.description_file_with_path
        if description_file is not None:
            release_paths.append(description_file)

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
        """Raw 'test' value."""
        return self._yaml.get('test', {})

    @property
    def _test_pytest(self) -> Dict[str, object]:
        """Raw 'test.pytest' value."""
        return self._test.get('pytest', {})

    @property
    def pytest_enabled(self) -> bool:
        """Whether pytests are enabled for the asset."""
        return self._test_pytest.get('enabled', False)

    @property
    def pytest_conda_environment(self) -> Path:
        """Conda environment definition for pytest."""
        return self._test_pytest.get('conda_environment')

    @property
    def pytest_conda_environment_with_path(self) -> Path:
        """Conda environment definition for pytest, appended to parent directory of asset config."""
        conda_environment = self.pytest_conda_environment
        return self._append_to_file_path(conda_environment) if conda_environment else None

    @property
    def pytest_pip_requirements(self) -> Path:
        """Pip requirements file for pytest."""
        return self._test_pytest.get('pip_requirements')

    @property
    def pytest_pip_requirements_with_path(self) -> Path:
        """Pip requirements file for pytest, appended to parent directory of asset config."""
        pip_requirements = self.pytest_pip_requirements
        return self._append_to_file_path(pip_requirements) if pip_requirements else None

    @property
    def pytest_tests_dir(self) -> Path:
        """Directory containing pytest scripts."""
        return self._test_pytest.get('tests_dir', ".") if self.pytest_enabled else None

    @property
    def pytest_tests_dir_with_path(self) -> Path:
        """Directory containing pytest scripts, appended to parent directory of asset config."""
        tests_dir = self.pytest_tests_dir
        return self._append_to_file_path(tests_dir) if tests_dir else None
