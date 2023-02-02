# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Create tags for released assets."""

import argparse
from git import Repo
from pathlib import Path

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger


def tag_released_assets(input_directory: Path,
                        asset_config_filename: str,
                        release_directory_root: Path,
                        git_username: str = None,
                        git_email: str = None):
    """Create tags for released assets.

    Args:
        input_directory (Path): Directory containing released assets.
        asset_config_filename (str): Asset config filename to search for.
        release_directory_root (Path): Release directory location.
        git_username (str, optional): User name to use for tag creation. Defaults to None.
        git_email (str, optional): Email address to use for tag creation. Defaults to None.
    """
    repo = Repo(release_directory_root)

    # Set username and email
    if git_username is not None:
        repo.config_writer().set_value("user", "name", git_username).release()
    if git_email is not None:
        repo.config_writer().set_value("user", "email", git_email).release()

    # Create tags locally
    tag_refs = []
    for asset_config in util.find_assets(input_directory, asset_config_filename):
        tag = asset_config.full_name
        message = f"Release {asset_config}"

        logger.print(f"Creating tag {tag}")
        tag_refs.append(repo.create_tag(tag, message=message))

    # Push tags
    for tag_ref in tag_refs:
        logger.print(f"Pushing tag {tag_ref}")
        repo.remotes.origin.push(tag_ref).raise_if_error()


if __name__ == "__main__":
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-directory", required=True, type=Path,
                        help="Directory containing released assets")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME,
                        help="Asset config file name to search for")
    parser.add_argument("-r", "--release-directory", required=True, type=Path,
                        help="Directory to which the release branch has been cloned")
    parser.add_argument("-u", "--username", help="Username for git push")
    parser.add_argument("-e", "--email", help="Email for git push")
    args = parser.parse_args()

    tag_released_assets(input_directory=args.input_directory,
                        asset_config_filename=args.asset_config_filename,
                        release_directory_root=args.release_directory,
                        git_username=args.username,
                        git_email=args.email)
