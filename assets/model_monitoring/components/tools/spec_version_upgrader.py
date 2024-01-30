# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class file for SpecVersionUpgrader and its helper class Spec."""

from typing import List
import os
import yaml
import re
from packaging.version import Version
from argparse import ArgumentParser


class Spec:
    """Spec class, represents a component spec."""

    def __init__(self, name: str, version: str, references: dict = None, path: str = None,
                 need_to_update: bool = False):
        """Init Spec instance."""
        self.name = name
        self.version = Version(version)
        self.references = references
        self.path = path
        self.need_to_update = need_to_update

    def __str__(self):
        """Get string representation of Spec."""
        return f"name:{self.name}, version:{self.version}, need_to_update:{self.need_to_update}"

    def __repr__(self):
        """Get string representation of Spec."""
        return self.__str__()

    def __eq__(self, other):
        """Compare two Spec instances."""
        if not other:
            return False
        if not isinstance(other, Spec):
            return False
        return self.name == other.name and self.version == other.version and self.references == other.references \
            and self.need_to_update == other.need_to_update and self.path == other.path

    @classmethod
    def from_reference(cls, reference: str) -> "Spec":
        """Get reference spec from reference string."""
        ref_pattern = r"^azureml://registries/azureml/components/(?P<name>[^/]+)/versions/(?P<version>[^/]+)$"
        match = re.match(ref_pattern, reference)
        if not match:
            raise ValueError(f"Invalid reference {reference}")

        return cls(match.group("name"), match.group("version"))

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "Spec":
        """Init spec from yaml file."""
        with open(yaml_file, 'r') as f:
            spec_dict = yaml.safe_load(f)
        references = {}
        jobs = spec_dict.get("jobs")
        if jobs:
            for step in jobs.values():
                refer_spec = Spec.from_reference(step["component"])
                references[refer_spec.name] = refer_spec.version

        return cls(spec_dict["name"], spec_dict["version"], references, yaml_file)


class SpecVersionUpgrader:
    """Spec version upgrader class, upgrade spec version in spec yaml files."""

    def __init__(self, spec_folder: str, spec_dict: dict = None):
        """Init SpecVersionUpgrader instance."""
        self._spec_dict = spec_dict or SpecVersionUpgrader._scan_spec_folder(spec_folder)

    def upgrade_versions(self, specs: List[str]):
        """Upgrade version of all specs in `spec_names`, including the specs that refer to the given specs."""
        # upgrade in spec_dict
        self._upgrade_versions(specs)
        # upgrade in real spec yaml files per spec_dict
        for spec in self._spec_dict.values():
            if spec.need_to_update:
                self._update_spec_yaml2(spec)

    def _update_spec_yaml(self, spec: Spec):
        """Update spec yaml file with new content."""
        # Note: this will re-order the yaml properties, please use _update_spec_yaml2 instead
        with open(spec.path, 'r') as f:
            spec_dict = yaml.safe_load(f)
        spec_dict["version"] = str(spec.version)
        if spec.references:
            for step in spec_dict["jobs"].values():
                refer_spec = Spec.from_reference(step["component"])
                if refer_spec.name in spec.references:
                    step["component"] = f"azureml://registries/azureml/components/{refer_spec.name}/versions/{spec.references[refer_spec.name]}"  # noqa: E501
        with open(spec.path, 'w') as f:
            yaml.safe_dump(spec_dict, f, indent=2)

    def _update_spec_yaml2(self, spec: Spec):
        """Update spec yaml with minor updates."""
        with open(spec.path, 'r') as f:
            yaml_content = f.read()
        # upgrade version
        yaml_content = re.sub(r"^version: \d+\.\d+\.\d+$", f"version: {spec.version}", yaml_content, 1, re.MULTILINE)
        # upgrade references
        if spec.references:
            for ref in spec.references:
                yaml_content = re.sub(
                    rf"(\s+)component: azureml://registries/azureml/components/{ref}/versions/\d+\.\d+\.\d+$",
                    rf"\1component: azureml://registries/azureml/components/{ref}/versions/{spec.references[ref]}",
                    yaml_content, flags=re.MULTILINE)

        with open(spec.path, 'w') as f:
            f.write(yaml_content)

    def _upgrade_versions(self, specs: List[str]):
        """Virtually upgrade version in spec_dict."""
        upgraded_specs = set()
        while specs:
            spec_name = specs.pop()
            if spec_name in upgraded_specs:
                # skip if spec already upgraded, this can happen if one subgraph refer multiple components
                continue
            parent_specs = self._upgrade_version_and_reference(spec_name)
            upgraded_specs.add(spec_name)
            specs.extend(parent_specs)

    def _upgrade_version_and_reference(self, spec_name: str) -> List[str]:
        """
        Upgrade spec version.

        Also for each of its parent specs(spec that has reference to this spec), update the reference version to be
        new version. Return all the parent specs that refer the upgraded spec.
        """
        # find the spec for spec_name, upgrade its version+1
        curr_spec = self._spec_dict.get(spec_name)
        if not curr_spec:
            raise ValueError(f"Spec {spec_name} not found")
        v = curr_spec.version
        curr_spec.version = Version(f"{v.major}.{v.minor}.{v.micro + 1}")
        curr_spec.need_to_update = True

        # upgrade reference version to the new version
        parent_specs = []
        for spec in self._spec_dict.values():
            if spec.references and spec.references.get(spec_name):
                spec.references[spec_name] = curr_spec.version
                spec.need_to_update = True
                parent_specs.append(spec.name)

        return parent_specs

    @staticmethod
    def _scan_spec_folder(spec_folder: str) -> dict:
        """Scan spec folder, return a dict of spec name to spec."""
        spec_dict = {}
        for root, _, files in os.walk(spec_folder):
            for file_name in files:
                if file_name == "spec.yaml":
                    spec = Spec.from_yaml(os.path.join(root, file_name))
                    spec_dict[spec.name] = spec

        return spec_dict


if __name__ == "__main__":
    # TODO add --all to upgrade all components
    argparser = ArgumentParser()
    argparser.add_argument("--specs", help="spec names to upgrade")
    args = argparser.parse_args()
    specs = args.specs.split(",") if args.specs else []
    spec_folder = os.path.abspath(f"{os.path.dirname(__file__)}/..")
    spec_version_upgrader = SpecVersionUpgrader(spec_folder)
    spec_version_upgrader.upgrade_versions(specs)
