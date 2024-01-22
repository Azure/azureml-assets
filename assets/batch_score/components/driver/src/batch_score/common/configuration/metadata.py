from argparse import Namespace
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .. import constants

@dataclass()
class Metadata(Namespace):
    component_name: str = field(init=True, default=None)
    component_version: str = field(init=True, default=None)

    @staticmethod
    def extract_component_name_and_version():
        try:
            _current_file_path = Path(os.path.abspath(__file__))
            metadata_file_path = _current_file_path.parents[2] / constants.METADATA_JSON_FILENAME

            with open(metadata_file_path, 'r') as json_file:
                component_metadata = json.load(json_file)
        except FileNotFoundError:
            print(f"The component metadata file '{metadata_file_path}' does not exist.")

        except Exception as e:
            print(f"An unexpected error occurred when extracting component name and version from metadata.json file: {e}")

        return Metadata.get_metadata(component_metadata)

    @staticmethod
    def get_metadata(metadata_payload: dict):
        component_name: str = metadata_payload.get(constants.COMPONENT_NAME_KEY, None)
        component_version: str = metadata_payload.get(constants.COMPONENT_VERSION_KEY, None)

        if component_name and (component_name.endswith(".yml") or component_name.endswith(".yaml")):
            component_name = component_name.rsplit('.', 1)[0]

        return Metadata(component_name=component_name, component_version=component_version)
