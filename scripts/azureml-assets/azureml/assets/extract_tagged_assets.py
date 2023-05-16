# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extract selected assets from release branch into output directory."""

import argparse
import re
from collections import defaultdict
from git import Repo
from pathlib import Path

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger

MATCHED_COUNT = "matched_count"
EXTRACTED_COUNT = "extracted_count"


def extract_tag_released_assets(release_directory_root: Path,
                                output_directory_root: Path,
                                pattern: re.Pattern = None,
                                include_deprecated: bool = False):
    """Extract selected assets from release branch and copy into output directory.

    Args:
        release_directory_root (Path): Release directory location.
        output_directory_root (Path): Output directory.
        pattern (re.Pattern, optional): Regex pattern for assets to extract and copy. Defaults to None.
        include_deprecated (bool, optional): Include deprecated assets. Defaults to False.
    """
    # Gather list of asset in release branch HEAD
    asset_list = set()
    if not include_deprecated:
        for asset_config in util.find_assets(input_dirs=release_directory_root):
            # Store as type/name
            asset_list.add(asset_config.partial_name)

    # Initialize counters
    matched_count = 0
    extracted_count = 0

    # Select tags
    repo = Repo(release_directory_root)
    commits_tags = defaultdict(list)
    for tag in repo.tags:
        if not pattern or pattern.fullmatch(tag.name):
            # Skip deprecated assets, if specified
            if not include_deprecated:
                type, name, _ = assets.AssetConfig.parse_full_name(tag.name)
                partial_name = f"{type.value}/{name}"
                if partial_name not in asset_list:
                    continue
            commits_tags[tag.commit].append(tag)
            matched_count += 1

    # Order commits for efficient checkout
    ordered_commits = sorted(commits_tags.keys(), key=lambda c: c.authored_datetime)

    # Checkout tags
    for commit in ordered_commits:
        tags = commits_tags[commit]
        logger.print(f"Checking out commit {commit} for {len(tags)} tag(s)")
        repo.git().checkout(commit)

        # Iterate over tags
        for tag in tags:
            # Copy asset to output directory
            type, name, version = assets.AssetConfig.parse_full_name(tag.name)
            release_dir = util.get_asset_release_dir_from_parts(type, name, release_directory_root)
            output_directory = util.get_asset_output_dir_from_parts(type, name, output_directory_root, version)
            if release_dir.exists():
                logger.print(f"Copying {type.value} {name}:{version} to {output_directory}")
                util.copy_replace_dir(release_dir, output_directory)
                extracted_count += 1
            else:
                logger.log_warning(f"{type.value.capitalize()} {name}:{version} not found in commit")

    # Reset to release branch
    repo.git().checkout("release")

    # Set variables
    logger.set_output(MATCHED_COUNT, matched_count)
    logger.set_output(EXTRACTED_COUNT, extracted_count)


if __name__ == "__main__":
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--release-directory", required=True, type=Path,
                        help="Directory to which the release branch has been cloned")
    parser.add_argument("-o", "--output-directory", required=True, type=Path,
                        help="Directory to which unreleased assets will be written")
    parser.add_argument("-t", "--pattern", type=re.compile,
                        help="Regex pattern to select assets to extract, in the format <type>/<name>/<version>")
    parser.add_argument("-d", "--include-deprecated", action="store_true",
                        help="Include deprecated assets")
    args = parser.parse_args()

    extract_tag_released_assets(release_directory_root=args.release_directory,
                                output_directory_root=args.output_directory,
                                pattern=args.pattern,
                                include_deprecated=args.include_deprecated)
