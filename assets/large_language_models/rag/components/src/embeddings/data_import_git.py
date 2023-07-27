# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""File for the Git Data Import.

Created on Feb 16, 2023

@author: nirovins
"""
import argparse
import os
import re
import shutil
import tempfile

from git import Repo


def _walk_and_copy(source: str, destination: str, wildcard: str):
    """Walk the directory and copy files.

    :param source: The source directory.
    :param destination: The destination directory.
    :param wildcard: The regex pattern to filter files.
    """
    search_pattern = re.compile(wildcard.lower())
    print("Start copying files.")
    file_count = 0
    for root, _, files in os.walk(source):
        # Do not parse, .git
        if os.path.split(root)[-1].startswith(".git"):
            continue
        for src_file in filter(lambda x: search_pattern.match(x.lower()), files):
            source_file = os.path.join(root, src_file)
            # We will prefix the file name with the path to the file in github repo
            # where the separator will be replaced by _. We have to add 1 to the source
            # because it does not contain the path separator.
            rel_path = source_file[len(source) + 1:]
            full_dest_path = os.path.join(destination, rel_path)
            os.makedirs(os.path.dirname(full_dest_path), exist_ok=True)
            if search_pattern.match(src_file.lower()):
                shutil.copy(
                    source_file,
                    full_dest_path,
                )
                file_count += 1
    print(f"{file_count} copied.")


def generate_data_from_git_repository(git_url: str, output_data: str, wildcard: str):
    """Get all files from the git repository, matching the wildcard.

    :param git_url: The git address to be cloned.
    :param output_data: The directory, where the data will be stored.
    :param wildcard: The regex pattern to filter files.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Cloning repository...")
        Repo.clone_from(git_url,
                        to_path=temp_dir)
        print("Repository cloned.")
        _walk_and_copy(temp_dir, output_data, wildcard)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-repository", type=str, required=True, dest='git_repository')
    parser.add_argument("--output-data", type=str, required=True, dest='output_data')
    parser.add_argument("--wildcard", type=str, required=True, dest='wildcard')
    username = os.environ.get("GIT_USERNAME")
    password = os.environ.get("GIT_PASSWORD")
    args = parser.parse_args()
    git_url = args.git_repository.split("//")[1]
    url_to_clone = f"https://{username}:{password}@{git_url}"
    generate_data_from_git_repository(url_to_clone, args.output_data, args.wildcard)
