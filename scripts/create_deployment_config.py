import argparse
import json
import os
import pygit2
from pathlib import Path

from config import AssetConfig, AssetType, EnvironmentConfig, PublishLocation, Spec
from ci_logger import logger

ENV_DEF_FILE_TEMPLATE = "envs/{name}/envdef.json"


def get_repo_remote_url(release_directory_root: str) -> str:
    repo = pygit2.Repository(release_directory_root)
    return repo.remotes['origin'].url


def get_repo_commit_hash(release_directory_root: str) -> str:
    repo = pygit2.Repository(release_directory_root)
    return repo[repo.head.target].hex


def create_deployment_config(input_directory: str,
                             asset_config_filename: str,
                             release_directory_root: str,
                             deployment_config_file_path: str):
    deployment_config = {}
    # Create in memory from contents of deployment config
    for root, _, files in os.walk(input_directory):
        for asset_config_file in [f for f in files if f == asset_config_filename]:
            # Load config
            asset_config = AssetConfig(os.path.join(root, asset_config_file))

            # Skip if not environment
            if asset_config.type is not AssetType.ENVIRONMENT:
                continue
            env_config = EnvironmentConfig(asset_config.extra_config_with_path)

            # Skip if not publishing to MCR
            if env_config.publish_location != PublishLocation.MCR:
                logger.log_warning(f"Skipping {asset_config.name} because it's not published to MCR")
                continue

            # Add to deployment config
            spec = Spec(asset_config.spec_with_path)
            env_def_file = ENV_DEF_FILE_TEMPLATE.format(name=asset_config.name)
            deployment_config[asset_config.name] = {
                'version': asset_config.version,
                'path': env_def_file,
                'contact': [],
                'metadata': {
                    'attributes': env_config.environment_metadata,
                    'tags': spec.tags,
                },
                'visible': env_config.environment_visible,
                'description': spec.description,
                'publish': {
                    'fullImageName': spec.image,
                },
                'imageDetails': None,
            }

            # Create envdef.json file
            remote_url = get_repo_remote_url(release_directory_root)
            commit_hash = get_repo_commit_hash(release_directory_root)
            build_context_path = Path(os.path.relpath(root, input_directory)).as_posix()

            git_url = f"{remote_url}#{commit_hash}:{build_context_path}"
            env_def = {
                'name': asset_config.name,
                'python': {
                    'userManagedDependencies': True
                },
                'docker': {
                    'buildContext': {
                        'locationType': 'git',
                        'location': git_url,
                        'dockerfilePath': env_config.dockerfile
                    },
                    'platform': {
                        'os': env_config.os.value.title(),
                        'architecture': "amd64"
                    }
                }
            }

            # Store environment definition file
            env_def_file_path = os.path.join(os.path.dirname(deployment_config_file_path), env_def_file)
            os.makedirs(os.path.dirname(env_def_file_path), exist_ok=True)
            with open(env_def_file_path, 'w') as f:
                json.dump(env_def, f, indent=4)

    # Create deployment config file
    with open(deployment_config_file_path, 'w') as f:
        json.dump(deployment_config, f, indent=4)
    print(f"Created deployment config file at {deployment_config_file_path}")


if __name__ == "__main__":
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-directory", required=True, help="Directory containing environment assets")
    parser.add_argument("-a", "--asset-config-filename", default="asset.yaml", help="Asset config file name to search for")
    parser.add_argument("-r", "--release-directory", required=True, help="Directory to which the release branch has been cloned")
    parser.add_argument("-o", "--deployment-config", required=True, help="Path to deployment config file")
    args = parser.parse_args()

    create_deployment_config(input_directory=args.input_directory,
                             asset_config_filename=args.asset_config_filename,
                             release_directory_root=args.release_directory,
                             deployment_config_file_path=args.deployment_config)
