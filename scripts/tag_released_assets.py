import argparse
import os
from git import Repo

from config import AssetConfig
from update_assets import get_release_tag_name


def tag_released_assets(input_directory: str,
                        asset_config_filename: str,
                        release_directory_root: str):
    repo = Repo(release_directory_root)

    # Create tags locally
    tag_refs = []
    for root, _, files in os.walk(input_directory):
        for asset_config_file in [f for f in files if f == asset_config_filename]:
            asset_config = AssetConfig(os.path.join(root, asset_config_file))
            tag = get_release_tag_name(asset_config)
            message = f"Release {asset_config.type.value} {asset_config.name} {asset_config.version}"

            print(f"Creating tag {tag}")
            tag_refs.append(repo.create_tag(tag, message=message))

    # Push tags
    for tag_ref in tag_refs:
        print(f"Pushing tag {tag_ref}")
        repo.remotes.origin.push(tag_ref).raise_if_error()


if __name__ == "__main__":
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-directory", required=True, help="Directory containing released assets")
    parser.add_argument("-a", "--asset-config-filename", default="asset.yaml", help="Asset config file name to search for")
    parser.add_argument("-r", "--release-directory", required=True, help="Directory to which the release branch has been cloned")
    args = parser.parse_args()

    tag_released_assets(input_directory=args.input_directory,
                        asset_config_filename=args.asset_config_filename,
                        release_directory_root=args.release_directory)
