# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper class for testing and maintaining consistency for LLM Components."""

from typing import List, Dict, Any, Tuple
import os
import re
import yaml


class Component():
    """Base Component Class."""

    def __init__(
            self,
            registry: str,
            name: str,
            version: str,
            environment: str = '',
            version_file: str = ''
            ):
        """Initialize the class."""
        self.registry = registry
        self.name = name
        self.version = version
        self.version_file = version_file
        self.environment = environment

    def __read_file__(self) -> Any:
        """Read the component text from file."""
        with open(self.version_file, 'r') as file:
            return file.read()


class Pipeline(Component):
    """Component Pipeline Class."""

    def __init__(
            self,
            registry: str,
            name: str,
            version: str,
            version_file: str = '',
            components: List[Component] = []
            ):
        """Initialize the class."""
        super(Pipeline, self).__init__(registry, name, version, version_file=version_file)
        self.components = components


def is_environment_valid(env_value: str) -> bool:
    """Ensure all environments are using the latest tag and not specific versions."""
    match = re.findall(
        r'azureml://registries/([^/]+)/environments/([^/]+)/versions/([\d\.]+)',
        env_value,
        flags=re.IGNORECASE)

    if len(match) <= 0:
        match = re.findall(
            r'([^:]+):([^\:\@]+)[\:\@]([^\:\@]+)',
            env_value,
            flags=re.IGNORECASE)

    return str(match[0][2]) == "latest"


def parse_component(comp_ref: str) -> Component:
    """Get the registry, name, and version from the component str."""
    match = re.findall(
        r'azureml://registries/([^/]+)/components/([^/]+)/versions/([\d\.]+)',
        comp_ref,
        flags=re.IGNORECASE)

    if len(match) <= 0:
        match = re.findall(
            r'([^:]+):([^:]+):([\.\d]+)',
            comp_ref,
            flags=re.IGNORECASE)

    return Component(match[0][0], match[0][1], str(match[0][2]))


def parse_jobs(jobs_value: str) -> List[Component]:
    """Pull the sub components from the Pipeline component."""
    deps = []
    for job_name in jobs_value:
        if 'component' in jobs_value[job_name]:
            deps.append(parse_component(jobs_value[job_name]['component']))
    return deps


target_fields = {
    'version': lambda x: str(x),
    'name': lambda x: x,
    'jobs': parse_jobs,
    'environment': lambda x: x,
}


skip_dirs = [
    "oai_v2_1p",  # owned by oai team
]


def generate_assets(folder_path: List[str]) -> Tuple[Dict[str, Component], Dict[str, Pipeline]]:
    """Walk the file tree and pull out all the components and component pipelines."""
    comp_assets_all = {}
    pipe_assets_all = {}
    for root, _, files in os.walk(folder_path):
        if [d for d in skip_dirs if d in root]:
            continue
        comp = parse_asset_yamls(files, root)
        if comp is not None:
            if isinstance(comp, Pipeline):
                pipe_assets_all[comp.name] = comp
            else:
                comp_assets_all[comp.name] = comp
    return (comp_assets_all, pipe_assets_all)


def parse_yaml(file_path: str) -> Any:
    """Read a yaml file."""
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


def parse_asset_yamls(files: List[str], root: str) -> Component:
    """Read a yaml file and ."""
    comp = {}
    for file in files:
        file_path = os.path.join(root, file)

        if file.lower().startswith('asset.y'):
            yml = parse_yaml(file_path)
            comp['type'] = yml['type']
        elif file.lower().startswith('spec.y'):
            yml = parse_yaml(file_path)
        else:
            continue

        for field in target_fields:
            if field in yml and (type(yml[field]) is not str or r'{{' not in yml[field]):
                extractor = target_fields[field]
                comp[field] = extractor(yml[field])

                if field == 'version':
                    comp['version_file'] = file_path

    if len(comp) == 0:
        return None

    if comp['type'] == 'component':
        jobs = comp['jobs'] if 'jobs' in comp else []
        if jobs is None or len(jobs) <= 0:
            env = comp.get('environment', None)
            return Component('', comp['name'], comp['version'], env, comp['version_file'])
        else:
            return Pipeline('', comp['name'], comp['version'], comp['version_file'], jobs)

    if comp['type'] == 'environment':
        return None

    raise Exception(f'Oops - found type {comp["type"]}')
