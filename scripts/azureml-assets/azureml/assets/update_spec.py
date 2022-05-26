import argparse
from pathlib import Path

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger


def update(asset_config: assets.AssetConfig, output_file: Path = None, version: str = None):
    """Update template tags in an asset's spec file using data from the asset config and any extra configs.

    Args:
        asset_config (assets.AssetConfig): AssetConfig object
        output_file (str, optional): File to which updated spec file will be written. If unspecified, the original spec file will be updated.
        version (str, optional): Version to use instead of the one in the asset config file.
    """
    # Create data
    data = {
        'asset': {
            'name': asset_config.name,
            'version': version or asset_config.version,
        }
    }

    # Augment with with type-specific data
    if asset_config.type == assets.AssetType.ENVIRONMENT:
        environment_config = assets.EnvironmentConfig(asset_config.extra_config_with_path)
        if environment_config.publish_location == assets.PublishLocation.MCR:
            data['image'] = {
                'name': environment_config.image_name,
                'publish': {
                    'hostname': environment_config.publish_location_hostname
                }
            }

    # Load spec template and render
    with open(asset_config.spec_with_path) as f:
        contents = util.render(f.read(), data)

    # Write spec
    if output_file == "-":
        logger.print(contents)
    else:
        if output_file is None:
            output_file = asset_config.spec_with_path
        with open(output_file, "w") as f:
            f.write(contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--asset-config", required=True, type=Path, help="Asset config file that points to the spec file to update")
    parser.add_argument("-o", "--output", type=Path, help="File to which output will be written. Defaults to the original spec file if not specified.")
    args = parser.parse_args()

    update(args.asset_config, args.output)
