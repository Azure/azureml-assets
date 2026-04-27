# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Create tags for released assets."""

import argparse
import sys
from git import Repo
from pathlib import Path
from typing import Set

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger

TAGGED_COUNT = "tagged_count"
SKIPPED_COUNT = "skipped_count"


def _matches_compliant_image(asset_config: assets.AssetConfig, compliant_images: Set[str]) -> bool:
    """Check whether an environment asset's image matches any entry in the compliant set.

    Matching is intentionally flexible: a compliant image FQIN like
    ``registry.azurecr.io/ns/image:tag`` will match if the asset's image name
    (e.g. ``ns/image``) appears as a substring of any compliant entry.

    Args:
        asset_config (assets.AssetConfig): Asset config for an environment.
        compliant_images (Set[str]): Set of compliant FQINs from vulnerability scan.

    Returns:
        bool: True if the asset's image is in the compliant set.
    """
    if asset_config.type != assets.AssetType.ENVIRONMENT:
        return True

    try:
        env_config = asset_config.extra_config_as_object()
        image_name = env_config.image_name
    except Exception:
        return False

    if not image_name:
        return False

    for fqin in compliant_images:
        if image_name in fqin:
            return True
    return False


def tag_released_assets(input_directory: Path,
                        asset_config_filename: str,
                        release_directory_root: Path,
                        git_username: str = None,
                        git_email: str = None,
                        compliant_images: Set[str] = None):
    """Create tags for released assets.

    When *compliant_images* is provided, only environment assets whose image
    name appears in that set will be tagged (fail-closed: environments not in
    the set are skipped). Non-environment assets are always tagged.

    Args:
        input_directory (Path): Directory containing released assets.
        asset_config_filename (str): Asset config filename to search for.
        release_directory_root (Path): Release directory location.
        git_username (str, optional): User name to use for tag creation. Defaults to None.
        git_email (str, optional): Email address to use for tag creation. Defaults to None.
        compliant_images (Set[str], optional): Set of scan-compliant image FQINs.
            When provided, only matching environments are tagged.
    """
    repo = Repo(release_directory_root)

    # Set username and email
    if git_username is not None:
        repo.config_writer().set_value("user", "name", git_username).release()
    if git_email is not None:
        repo.config_writer().set_value("user", "email", git_email).release()

    tagged_count = 0
    skipped_count = 0

    # Create tags locally
    tag_refs = []
    for asset_config in util.find_assets(input_directory, asset_config_filename):
        # Gate environment tagging on vulnerability scan compliance
        if compliant_images is not None and not _matches_compliant_image(asset_config, compliant_images):
            logger.log_warning(f"Skipping tag for {asset_config.full_name}: image not in compliant list")
            skipped_count += 1
            continue

        tag = asset_config.full_name
        message = f"Release {asset_config}"

        logger.print(f"Creating tag {tag}")
        tag_refs.append(repo.create_tag(tag, message=message))
        tagged_count += 1

    # Push tags
    for tag_ref in tag_refs:
        logger.print(f"Pushing tag {tag_ref}")
        repo.remotes.origin.push(tag_ref).raise_if_error()

    logger.print(f"Tagged {tagged_count} asset(s), skipped {skipped_count} non-compliant asset(s)")
    logger.set_output(TAGGED_COUNT, tagged_count)
    logger.set_output(SKIPPED_COUNT, skipped_count)

    return skipped_count


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
    parser.add_argument("--compliant-images",
                        help="Comma-separated list of scan-compliant image FQINs. "
                             "When provided, only environments whose image matches "
                             "an entry in this list will be tagged (fail-closed).")
    args = parser.parse_args()

    # Parse compliant images into a set if provided
    compliant_set = None
    if args.compliant_images is not None:
        compliant_set = set(img.strip() for img in args.compliant_images.split(",") if img.strip())
        if not compliant_set:
            logger.log_error("--compliant-images was provided but is empty. "
                             "Refusing to tag any assets (fail-closed).")
            sys.exit(1)
        logger.print(f"Compliant images filter active: {len(compliant_set)} image(s)")

    skipped = tag_released_assets(input_directory=args.input_directory,
                                  asset_config_filename=args.asset_config_filename,
                                  release_directory_root=args.release_directory,
                                  git_username=args.username,
                                  git_email=args.email,
                                  compliant_images=compliant_set)
