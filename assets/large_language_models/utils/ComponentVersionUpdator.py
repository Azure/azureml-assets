# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

r"""
Helper File to Update Large Language Model Versions.

Utility File for incrementing, synchronizing, and updating all LLM versions.
Also checks Environments are tagging latest.

To update all:
\\azureml-assets\assets\large_language_models>
python utils\ComponentVersionUpdator.py --input_folder . --update_all

To only update component pipelines with updated components:
\\azureml-assets\assets\large_language_models>
python utils\ComponentVersionUpdator.py --input_folder .
"""

from typing import List, Dict, Any, Tuple
import os
import re
import yaml


class Component():
    """Base Component Class."""

    def __init__(self, registry: str, name: str, version: str, version_file: str = ''):
        """Initialize the class."""
        self.registry = registry
        self.name = name
        self.version = version
        self.version_file = version_file

    def __read_file__(self) -> Any:
        """Read the component text from file."""
        with open(self.version_file, 'r') as file:
            return file.read()

    def __write_file__(self, yaml_data) -> None:
        """Write the updated component text to file."""
        with open(self.version_file, 'w') as outfile:
            outfile.write(yaml_data)

    def increment_component(self):
        """Get and increment the component version."""
        old_version = self.version
        v_parts = [int(x) for x in self.version.split('.')]
        v_parts[-1] += 1
        self.version = ".".join([str(x) for x in v_parts])

        text_data = self.__read_file__()

        text_data = text_data.replace(
            f'version: {old_version}',
            f'version: {self.version}'
        )

        self.__write_file__(text_data)


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
        super(Pipeline, self).__init__(registry, name, version, version_file)
        self.components = components

    def increment_component(self, comp_assets_all: Dict[str, Component], update_all: bool):
        """Get and increment the component version, then update all the child components."""
        needs_update = update_all

        old_version = self.version
        v_parts = [int(x) for x in self.version.split('.')]
        v_parts[-1] += 1
        self.version = ".".join([str(x) for x in v_parts])

        text_data = self.__read_file__()

        text_data = text_data.replace(
            f'version: {old_version}',
            f'version: {self.version}'
        )

        yaml_data = yaml.safe_load(text_data)

        for s_c in yaml_data['jobs']:
            sub_comp_str = yaml_data['jobs'][s_c]['component']
            sub_comp = parse_component(sub_comp_str)

            try:
                new_version = comp_assets_all[sub_comp.name].version
            except KeyError as e:
                print(f"[{self.name}] Cannot find sub-component {e}, assuming that it comes from a skipped directory.")
                continue

            if sub_comp.version != new_version:
                needs_update = True
                new_version_str = sub_comp_str.replace(
                    sub_comp.version,
                    new_version)

                text_data = text_data.replace(
                    sub_comp_str,
                    new_version_str
                )

        if needs_update:
            self.__write_file__(text_data)


def main_wrapper(folder_path: str, update_all: bool) -> None:
    """Execute the code."""
    (comp_assets_all, pipe_assets_all) = generate_assets(folder_path)

    if update_all:
        print(f'Updating component versions (Count:{len(comp_assets_all)})....')
        for a in comp_assets_all:
            print('  ', a)
            comp_assets_all[a].increment_component()
    else:
        print(f'Skipping update component versions (Count:{len(comp_assets_all)})....')

    print(f'Updating Pipeline and linked Component versions (Count:{len(pipe_assets_all)})....')
    for a in pipe_assets_all:
        print('  ', a)
        pipe_assets_all[a].increment_component(comp_assets_all, update_all)


def validate_environment(compt_name: str, env_value: str) -> None:
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

    if str(match[0][2]) != "latest":
        print('Error', compt_name, 'environment', match[0][1], 'is not using the latest tag')


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
    'jobs': parse_jobs
}

validate_fields = {
    'environment': validate_environment
}

skip_dirs = [
    "oai_v2_1p",  # owned by oai team
    "prompts",
    "dbcopilot"
]


def generate_assets(folder_path: List[str]) -> Tuple[Dict[str, Component], Dict[str, Pipeline]]:
    """Walk the file tree and pull out all the components and component pipelines."""
    comp_assets_all = {}
    pipe_assets_all = {}
    for root, dirs, files in os.walk(folder_path, topdown=True):
        # https://stackoverflow.com/questions/19859840/excluding-directories-in-os-walk
        new_dirs = []
        for d in dirs:
            if d not in skip_dirs:
                new_dirs.append(d)
            else:
                print(f"Going to skip dir '{d}' in root '{root}'")
        dirs[:] = new_dirs
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
            if field in yml and (not isinstance(yml[field], str) or r'{{' not in yml[field]):
                extractor = target_fields[field]
                comp[field] = extractor(yml[field])

                if field == 'version':
                    comp['version_file'] = file_path

        for field in validate_fields:
            if field in yml and (not isinstance(yml[field], str) or r'{{' not in yml[field]):
                validate_fields[field](comp['name'], yml[field])

    if len(comp) == 0:
        return None

    if comp['type'] == 'component':
        jobs = comp['jobs'] if 'jobs' in comp else []
        if jobs is None or len(jobs) <= 0:
            return Component('', comp['name'], comp['version'], comp['version_file'])
        else:
            return Pipeline('', comp['name'], comp['version'], comp['version_file'], jobs)

    if comp['type'] == 'environment':
        return None

    raise Exception('Oops')


if __name__ == '__main__':
    from argparse import ArgumentParser, BooleanOptionalAction

    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str, default='.')
    parser.add_argument('--update_all', action=BooleanOptionalAction, required=False, default=False)
    args = parser.parse_args()

    main_wrapper(args.input_folder, args.update_all)
