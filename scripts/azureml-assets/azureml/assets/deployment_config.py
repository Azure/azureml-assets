# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deployment config classes."""

from dataclasses import dataclass
from typing import Dict, List
import azureml.assets as assets
from marshmallow import Schema, fields, post_load, validates, validates_schema
from ruamel.yaml import YAML

"""
Sample deplyment config YAML file:

create: # Assets to create
  component: # List of components
    - component1
  environment: # List of environments
    - environment1

update: # Assets to update
  components: # Not yet supported
  environment: # Dictionary of environments
    acpt-pytorch-1.11-cuda11.5: # Environment name
      # List containing updates to make, organized by versions to affect
      - versions: ["1", "2"] # Specific versions to update; don't use with all_versions
        all_versions: true # Update all versions; don't use with versions
        description: "DEPRECATED: Please use acpt-pytorch-1.11-cuda11.3 instead." # Replaces any existing description
        tags: # Tag updates
          add: # Add or update existing tags
            Tag1: "value"
            Deprecated: "" # Tags with no values are okay
          replace: # Replace any existing tags with this set; don't use with add or delete
            Tag1: "value"
            Tag2: "value"
          delete: # Tags to delete; delete wins any overlap with add
            - Tag1
            - Tag2
        stage: "Archived" # Use Active or Archived to control visibility to list operations

delete: # Assets to delete
  component: # List of components
    name: microsoft_azureml_automl_classification_component
      - versions: ["2"] # Specific versions to delete; don't use with all_versions
        all_versions: true # Deletes all versions
        delete_container: true # Also delete asset container
"""


@dataclass
class AssetTags:
    """Asset tags class.

    Args:
        add (Dict[str, str]): Tags to add.
        replace (Dict[str, str]): Replace any existing tags with these.
        delete (List[str]): Tags to delete.
    """

    add: Dict[str, str] = None
    replace: Dict[str, str] = None
    delete: List[str] = None


@dataclass
class AssetSystemMetadata:
    """Asset system metadata class.

    Args:
        add (Dict[str, str]): System metadata to add.
        replace (Dict[str, str]): Replace any existing system metadata with these.
        delete (List[str]): System metadata to delete.
    """

    add: Dict[str, str] = None
    replace: Dict[str, str] = None
    delete: List[str] = None


@dataclass
class AssetProperties:
    """Asset properties class.

    Args:
        add (Dict[str, str]): Properties to add.
    """

    add: Dict[str, str] = None


@dataclass
class Versions:
    """Versions base class.

    Args:
        versions (List[str]): Versions to affect.
        all_versions (bool): Affect all versions.
    """

    versions: List[str] = None
    all_versions: bool = False


@dataclass
class AssetVersionUpdate(Versions):
    """Asset version update class.

    Args:
        description (str): New description.
        tags (AssetTags): Tag updates.
        properties (AssetProperties): Property updates.
        stage (str): New stage.
        system_metadata (AssetSystemMetadata): System metadata updates.
    """

    description: str = None
    tags: AssetTags = None
    properties: AssetProperties = None
    stage: str = None
    system_metadata: AssetSystemMetadata = None

    def __post_init__(self):
        """Convert field values to objects."""
        if self.tags:
            self.tags = AssetTags(**self.tags)

        if self.properties:
            self.properties = AssetProperties(**self.properties)

        if self.system_metadata:
            self.system_metadata = AssetSystemMetadata(**self.system_metadata)


@dataclass
class AssetUpdate:
    """Asset update class.

    Args:
        name (str): Asset name.
        updates (List[AssetVersionUpdate]): Updates to apply.
    """

    name: str = None
    updates: List[AssetVersionUpdate] = None

    def __post_init__(self):
        """Convert field values to objects."""
        if self.updates:
            self.updates = [AssetVersionUpdate(**u) for u in self.updates]


@dataclass
class AssetVersionDelete(Versions):
    """Asset version delete class.

    Args:
        delete_container (bool): Delete container.
    """

    delete_container: bool = False


@dataclass
class AssetDelete:
    """Asset delete class.

    Args:
        name (str): Asset name.
        deletes (List[AssetVersionDelete]): Deletes to apply.
    """

    name: str = None
    deletes: List[AssetVersionDelete] = None

    def __post_init__(self):
        """Convert field values to objects."""
        if self.deletes:
            self.deletes = [AssetVersionDelete(**d) for d in self.deletes]


@dataclass
class DeploymentConfig:
    """Deployment config class.

    Args:
        create (Dict[assets.AssetType, List[str]]): Assets to create.
        update (Dict[assets.AssetType, List[AssetUpdate]]): Assets to update.
        delete (Dict[assets.AssetType, List[AssetDelete]]): Assets to delete.
    """

    create: Dict[assets.AssetType, List[str]] = None
    update: Dict[assets.AssetType, List[AssetUpdate]] = None
    delete: Dict[assets.AssetType, List[AssetDelete]] = None

    def __post_init__(self):
        """Convert field values to objects."""
        if self.update:
            self.update = {k: self._convert_update_dict(v) for k, v in self.update.items()}
        if self.delete:
            self.delete = {k: self._convert_delete_dict(v) for k, v in self.delete.items()}

    @staticmethod
    def _convert_update_dict(update_assets: Dict[str, object]) -> List[AssetUpdate]:
        return [AssetUpdate(name=k, updates=v) for k, v in update_assets.items()]

    @staticmethod
    def _convert_delete_dict(delete_assets: Dict[str, object]) -> List[AssetDelete]:
        return [AssetDelete(name=k, deletes=v) for k, v in delete_assets.items()]

    @staticmethod
    def load(deployment_config: str) -> "DeploymentConfig":
        """Load a deployment config from a file.

        Args:
            deployment_config (str): Deployment config file.

        Returns:
            DeploymentConfig: Deployment config.
        """
        with open(deployment_config) as fp:
            config = YAML().load(fp)
            return DeploymentConfigSchema().load(config)

    def should_create(self, asset_type: assets.AssetType, asset_name: str) -> bool:
        """Determine if an asset should be created.

        Args:
            asset_type (assets.AssetType): Asset type.
            asset_name (str): Asset name.

        Returns:
            bool: True if the asset should be created.
        """
        return any(n in {"*", asset_name} for n in self.create.get(asset_type, [])) if self.create else False


class TagsSchema(Schema):
    """Tags schema."""

    add = fields.Dict(fields.Str(), fields.Str())
    replace = fields.Dict(fields.Str(), fields.Str())
    delete = fields.List(fields.Str())

    @validates('add')
    def _validate_add(self, value: Dict[str, str]):
        if value is not None and not value:
            raise ValueError("add must be non-empty")

    @validates('delete')
    def _validate_delete(self, value: List[str]):
        if value is not None and not value:
            raise ValueError("delete must be non-empty")

    @validates_schema
    def _validate_schema(self, data: Dict[str, object], **kwargs):
        if data.get('replace') and (data.get('add') or data.get('delete')):
            raise ValueError("replace can't be used with add or delete")


class SystemMetadataSchema(Schema):
    """System metadata schema."""

    add = fields.Dict(fields.Str(), fields.Str())
    replace = fields.Dict(fields.Str(), fields.Str())
    delete = fields.List(fields.Str())

    @validates('add')
    def _validate_add(self, value: Dict[str, str]):
        if value is not None and not value:
            raise ValueError("add must be non-empty")

    @validates('delete')
    def _validate_delete(self, value: List[str]):
        if value is not None and not value:
            raise ValueError("delete must be non-empty")

    @validates_schema
    def _validate_schema(self, data: Dict[str, object], **kwargs):
        if data.get('replace') and (data.get('add') or data.get('delete')):
            raise ValueError("replace can't be used with add or delete")


class PropertiesSchema(Schema):
    """Properties schema."""

    add = fields.Dict(fields.Str(), fields.Str())

    @validates('add')
    def _validate_add(self, value: Dict[str, str]):
        if value is not None and not value:
            raise ValueError("add must be non-empty")


class VersionsSchema(Schema):
    """Versions schema."""

    versions = fields.List(fields.Str())
    all_versions = fields.Boolean()

    @validates('versions')
    def _validate_versions(self, value: List[str]):
        if value is not None and not value:
            raise ValueError("versions must be non-empty")

    @validates_schema
    def _validate_schema(self, data: Dict[str, object], **kwargs):
        if data.get('versions') and data.get('all_versions'):
            raise ValueError("Only one of versions and all_versions can be specified")


class AssetVersionUpdateSchema(VersionsSchema):
    """Asset version update schema."""

    description = fields.Str()
    tags = fields.Nested(TagsSchema)
    properties = fields.Nested(PropertiesSchema)
    stage = fields.Str()
    system_metadata = fields.Nested(SystemMetadataSchema)


class AssetVersionDeleteSchema(VersionsSchema):
    """Asset version delete schema."""

    delete_container = fields.Boolean()


class DeploymentConfigSchema(Schema):
    """Deployment config schema."""

    create = fields.Dict(fields.Enum(assets.AssetType, by_value=True), fields.List(fields.Str()))
    update = fields.Dict(
        fields.Enum(assets.AssetType, by_value=True),
        fields.Dict(
            fields.Str(),
            fields.List(fields.Nested(AssetVersionUpdateSchema))
        )
    )
    delete = fields.Dict(
        fields.Enum(assets.AssetType, by_value=True),
        fields.Dict(
            fields.Str(),
            fields.List(fields.Nested(AssetVersionDeleteSchema))
        )
    )

    @post_load
    def _convert_to_object(self, data, **kwargs):
        # Convert to objects
        return DeploymentConfig(**data)
