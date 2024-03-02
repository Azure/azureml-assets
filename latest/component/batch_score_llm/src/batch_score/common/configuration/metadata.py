# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Component metadata."""

from argparse import Namespace
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .. import constants
from ...utils.local_utils import is_running_in_azureml_job


@dataclass()
class Metadata(Namespace):
    """Component metadata."""

    component_name: str = field(init=True, default=None)
    component_version: str = field(init=True, default=None)

    @staticmethod
    def extract_component_name_and_version():
        """Extract component name and version from metadata.json file."""
        try:
            _current_file_path = Path(os.path.abspath(__file__))
            metadata_file_path = _current_file_path.parents[2] / constants.METADATA_JSON_FILENAME

            with open(metadata_file_path, 'r') as json_file:
                component_metadata = json.load(json_file)
                return Metadata.get_metadata(component_metadata)

        except FileNotFoundError:
            print(f"The component metadata file '{metadata_file_path}' does not exist.")
            if is_running_in_azureml_job():
                raise  # A missing metadata file is treated as a fatal error when running in AzureML.
            else:
                return Metadata._dummy()

        except Exception as e:
            print("An unexpected error occurred when extracting component name and version "
                  f"from metadata.json file: {e}")
            if is_running_in_azureml_job():
                raise  # A missing metadata file is treated as a fatal error when running in AzureML.
            else:
                return Metadata._dummy()

    @staticmethod
    def get_metadata(metadata_payload: dict):
        """Get metadata object."""
        component_name: str = metadata_payload.get(constants.COMPONENT_NAME_KEY, None)
        component_version: str = metadata_payload.get(constants.COMPONENT_VERSION_KEY, None)

        if component_name and (component_name.endswith(".yml") or component_name.endswith(".yaml")):
            component_name = component_name.rsplit('.', 1)[0]

        return Metadata(component_name=component_name, component_version=component_version)

    @staticmethod
    def _dummy():
        """Get dummy metadata object."""
        return Metadata(component_name='component_name', component_version='0.0.0')
