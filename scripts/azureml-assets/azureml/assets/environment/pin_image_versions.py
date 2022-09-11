# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import re
import urllib.parse
from pathlib import Path
from typing import List, Tuple
from urllib.request import Request, urlopen

from azureml.assets.util import logger

LATEST_TAG = "latest"
# Handles registry/image_name:{{latest-image-tag}} and registry/image_name:{{latest-image-tag:regex}}
LATEST_IMAGE_TAG = re.compile(r"([^\"'\s]+):\{\{latest-image-tag(?::(.+))?\}\}")


def get_latest_tag_or_digest(image: str, tags: List[str]) -> Tuple[str, str]:
    (hostname, repo) = image.split("/", 1)

    # Find another tag corresponding to latest
    latest_tag = None
    latest_digest = None
    for tag in tags:
        # Retrieve digest
        encoded_tag = urllib.parse.quote(tag, safe="")
        request = Request(f"https://{hostname}/v2/{repo}/manifests/{encoded_tag}",
                          method="HEAD",
                          headers={'Accept': "application/vnd.docker.distribution.manifest.v2+json"})
        try:
            response = urlopen(request)
        except Exception as e:
            raise Exception(f"Failed to retrieve manifest for {repo}:{tag}: {e}")
        digest = response.info()['Docker-Content-Digest']

        if tag == LATEST_TAG:
            # Store latest digest for comparison
            latest_digest = digest
        elif digest == latest_digest:
            # Found matching digest
            latest_tag = tag
            break

    return latest_tag, latest_digest


def get_latest_image_suffix(image: str, regex: re.Pattern = None) -> str:
    (hostname, repo) = image.split("/", 1)

    # Retrieve tags
    try:
        response = urlopen(f"https://{hostname}/v2/{repo}/tags/list")
    except Exception as e:
        raise Exception(f"Failed to retrieve tags for {repo}: {e}")
    tags = json.loads(response.read().decode("utf-8")).get("tags", [])

    # Filter tags and sort in descending order because this should be faster
    tags_sorted = sorted([t for t in tags if t != LATEST_TAG and
                         (regex is None or regex.search(t) is not None)], reverse=True)

    # Handle regex
    if regex is not None:
        # Use the most recent matching tag
        if tags_sorted:
            latest_tag = tags_sorted[0]
        else:
            raise Exception(f"{image} does not have tag that matches {regex}")
    else:
        # Ensure latest is present
        if LATEST_TAG not in tags:
            raise Exception(f"{image} does not have a {LATEST_TAG} tag")

        # Insert latest at the beginning to ensure we get its digest first
        tags_sorted.insert(0, LATEST_TAG)

        # Find another tag corresponding to latest, or default to digest
        latest_tag, latest_digest = get_latest_tag_or_digest(image, tags)

    # Return tag or digest
    if latest_tag is not None:
        return f":{latest_tag}"
    else:
        logger.log_warning(f"Using digest for {image} because a non-{LATEST_TAG} was not found")
        return f"@{latest_digest}"


def pin_images(contents: str) -> str:
    # Process MCR template tags
    while True:
        match = LATEST_IMAGE_TAG.search(contents)
        if not match:
            break
        repo = match.group(1)
        regex = match.group(2)
        message = f"Finding latest image tag/digest for {repo}"
        if regex is not None:
            message += f" matching {regex}"
            regex = re.compile(regex)
        logger.log_debug(message)
        suffix = get_latest_image_suffix(repo, regex)
        logger.log_debug(f"Latest image reference is {repo}{suffix}")
        contents = contents[:match.start()] + f"{repo}{suffix}" + contents[match.end():]

    return contents


def transform_file(input_file: Path, output_file: Path = None):
    # Read file
    with open(input_file) as f:
        contents = f.read()

    # Transform
    contents = pin_images(contents)

    # Write to stdout or output_file
    if output_file == "-":
        logger.print(contents)
    else:
        if output_file is None:
            output_file = input_file
        with open(output_file, "w") as f:
            f.write(contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path,
                        help="File containing images to pin to latest versions", required=True)
    parser.add_argument("-o", "--output", type=Path,
                        help="File to which output will be written. Defaults to the input file.")
    args = parser.parse_args()

    output = args.output or args.input
    transform_file(args.input, output)
