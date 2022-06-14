import argparse
import json
import os
from pathlib import Path
from typing import Dict

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger

ENV_DEF_FILE_TEMPLATE = "{name}.json"
SPEC_FILE_TEMPLATE = "{name}.yaml"
GIT_URL_TEMPLATE = "{{asset.repo.url}}#{{asset.repo.commit_hash}}:{{asset.repo.build_context.path}}"


def create_ems_payload(asset_config: assets.AssetConfig, env_config: assets.EnvironmentConfig, spec: assets.Spec,
                       template_data: Dict[str, object], full_image_name: str):

    # Start environment definition's docker section
    docker_section = {
        'platform': {
            'os': env_config.os.value.title(),
            'architecture': "amd64"
        }
    }
    if env_config.build_enabled:
        # Add buildContext section to environment definition
        docker_section['buildContext'] = {
            'locationType': 'git',
            'location': util.render(GIT_URL_TEMPLATE, template_data),
            'dockerfilePath': env_config.dockerfile
        }
    else:
        # Use existing image name
        docker_section['baseImage'] = full_image_name
    
    # Create payload
    payload = {
        'metadata': {
            'tags': spec.tags,
            'description': spec.description,
            'attributes': env_config.environment_metadata,
        },
        'definition': {
            'name': asset_config.name,
            'python': {
                'userManagedDependencies': True
            },
            'docker': docker_section
        }
    }

    return payload


def create_mfe_payload(env_config: assets.EnvironmentConfig, spec: assets.Spec, template_data: Dict[str, object],
                       full_image_name: str):    
    # Start properties
    properties = {
        'description': spec.description,
        'osType': env_config.os.value,
        'properties': {
            'azureml.imagename': full_image_name
        }
    }
    if spec.tags:
        properties['tags'] = spec.tags

    # Add image details
    if env_config.build_enabled:
        # Add build section to payload
        properties['build'] = {
            'contextUri': util.render(GIT_URL_TEMPLATE, template_data),
            'dockerfilePath': env_config.dockerfile
        }
    else:
        # Use existing image name
        properties['image'] = full_image_name

    # Create payload
    payload = { 'properties': properties }

    return payload


def create_deployment_config(input_directory: Path,
                             asset_config_filename: str,
                             release_directory_root: Path,
                             deployment_config_file_path: Path,
                             envs_dirname: Path,
                             specs_dirname: Path = None,
                             version_template: str = None,
                             tag_template: str = None):
    deployment_config = {}
    for asset_config in util.find_assets(input_directory, asset_config_filename, assets.AssetType.ENVIRONMENT):
        env_config = asset_config.environment_config_as_object()
        spec = asset_config.spec_as_object()

        # Determine full image name
        full_image_name = env_config.get_full_image_name(asset_config.version)
        if env_config.build_enabled and env_config.publish_location == assets.PublishLocation.MCR:
            # Apply tag template to image name
            full_image_name = util.apply_tag_template(full_image_name, tag_template)
        else:
            # Use image name as-is
            logger.log_warning(f"Not applying tag template to {asset_config.name} because it's a pre-published image")

        # Apply version template
        version = util.apply_version_template(asset_config.version, version_template)

        # Add to deployment config
        env_def_file = envs_dirname / ENV_DEF_FILE_TEMPLATE.format(name=asset_config.name)
        config = {
            'version': version,
            'path': env_def_file.as_posix(),
            'publish': {
                'fullImageName': full_image_name,
            }
        }
        new_spec_file = None
        if specs_dirname:
            new_spec_file = specs_dirname / SPEC_FILE_TEMPLATE.format(name=asset_config.name)
            config['spec_path'] = new_spec_file.as_posix()
        deployment_config[asset_config.name] = config

        # Create template data, used to render git URL below
        data = assets.create_template_data(asset_config=asset_config, release_directory_root=release_directory_root,
                                           include_commit_hash=True)

        # Create payload
        env_def_with_metadata = create_mfe_payload(env_config=env_config, spec=spec, template_data=data,
                                                   full_image_name=full_image_name)

        # Store payload
        env_def_file_path = deployment_config_file_path.parent / env_def_file
        os.makedirs(env_def_file_path.parent, exist_ok=True)
        with open(env_def_file_path, 'w') as f:
            json.dump(env_def_with_metadata, f, indent=4)

        # Store spec file
        if new_spec_file:
            new_spec_file_path = deployment_config_file_path.parent / new_spec_file
            os.makedirs(new_spec_file_path.parent, exist_ok=True)
            assets.update_spec(asset_config=asset_config, output_file=new_spec_file_path, data=data)

    # Create deployment config file
    with open(deployment_config_file_path, 'w') as f:
        json.dump(deployment_config, f, indent=4)
    logger.print(f"Created deployment config file at {deployment_config_file_path}")


if __name__ == "__main__":
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-directory", required=True, type=Path, help="Directory containing environment assets")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME, help="Asset config file name to search for")
    parser.add_argument("-r", "--release-directory", required=True, type=Path, help="Directory to which the release branch has been cloned")
    parser.add_argument("-o", "--deployment-config", required=True, type=Path, help="Path to deployment config file")
    parser.add_argument("-e", "--environment-definitions-dir", default=Path("envs"), type=Path, help="Name of directory that will contain environment definitions")
    parser.add_argument("-s", "--spec-files-dirname", type=Path, help="Name of directory that will contain environment spec files")
    parser.add_argument("-v", "--version-template", help="Template to apply to versions from spec files "
                        "file, example: '{version}.dev1'")
    parser.add_argument("-T", "--tag-template", help="Template to apply to image name tags from spec files "
                        "config file, example: '{tag}.dev1'")
    args = parser.parse_args()

    create_deployment_config(input_directory=args.input_directory,
                             asset_config_filename=args.asset_config_filename,
                             release_directory_root=args.release_directory,
                             deployment_config_file_path=args.deployment_config,
                             envs_dirname=args.environment_definitions_dir,
                             specs_dirname=args.spec_files_dirname,
                             version_template=args.version_template,
                             tag_template=args.tag_template)
