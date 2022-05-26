import argparse
import json
from pathlib import Path

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger


def create_import_config(input_directory: Path,
                         asset_config_filename: str,
                         import_config_file_path: Path,
                         registry_address: str,
                         tag: str = None,
                         registry_username: str = None,
                         registry_password: str = None):
    images = []
    for asset_config in util.find_assets(input_directory, asset_config_filename, assets.AssetType.ENVIRONMENT):
        env_config = assets.EnvironmentConfig(asset_config.extra_config_with_path)

        # Skip if not publishing to MCR
        if env_config.publish_location != assets.PublishLocation.MCR:
            continue

        # Get source image name
        version = assets.Spec(asset_config.spec_with_path).version
        source_image_name = env_config.get_image_name_with_tag(version)

        # Form destination image name and tag
        destination_image_name = f"{env_config.publish_visibility.value}/{env_config.image_name}"
        image_tag = version

        # Generate target image names
        destination_tags = [tag] if tag is not None else [image_tag, "latest"]
        destination_images = [f"{destination_image_name}:{tag}" for tag in destination_tags]

        # Store image info
        source = {
            'imageName': f"{registry_address}/{source_image_name}",
        }
        if registry_username is not None and registry_password is not None:
            source['registryUsername'] = registry_username
            source['registryPassword'] = registry_password
        images.append({
            'source': source,
            'destination': {
                'imageNames': destination_images
            }
        })

    # Create on disk
    import_config = {'images': images}
    with open(import_config_file_path, 'w') as f:
        json.dump(import_config, f, indent=2)
    logger.print(f"Created import config file at {import_config_file_path}")


if __name__ == "__main__":
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-directory", required=True, type=Path, help="Directory containing environment assets")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME, help="Asset config file name to search for")
    parser.add_argument("-o", "--import-config", required=True, type=Path, help="Path to import config file")
    parser.add_argument("-A", "--registry-address", required=True, help="Address of registry login server")
    parser.add_argument("-t", "--tag", help="Destination tag to use instead of asset version and latest")
    parser.add_argument("-u", "--registry-username", help="Registry username")
    parser.add_argument("-p", "--registry-password", help="Registry password")
    args = parser.parse_args()

    # Ensure both username and password are specified, or neither
    if (args.registry_username is not None) ^ (args.registry_password is not None):
        parser.error("--registry-username or --registry-password can't be used alone.")

    create_import_config(input_directory=args.input_directory,
                         asset_config_filename=args.asset_config_filename,
                         import_config_file_path=args.import_config,
                         registry_address=args.registry_address,
                         tag=args.tag,
                         registry_username=args.registry_username,
                         registry_password=args.registry_password)
