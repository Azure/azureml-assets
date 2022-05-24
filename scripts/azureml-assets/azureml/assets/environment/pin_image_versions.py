import argparse
import json
import re
import urllib.parse
from pathlib import Path
from urllib.request import Request, urlopen

from azureml.assets.util import logger

LATEST_TAG = "latest"
LATEST_IMAGE_TAG = re.compile(r"([^\"'\s]+):\{\{latest-image-tag\}\}")


def get_latest_image_suffix(image: str):
    (hostname, repo) = image.split("/", 1)

    # Retrieve tags
    try:
        response = urlopen(f"https://{hostname}/v2/{repo}/tags/list")
    except Exception as e:
        raise Exception(f"Failed to retrieve tags for {repo}: {e}")
    tags = json.loads(response.read().decode("utf-8")).get("tags", [])

    # Ensure latest is present
    if LATEST_TAG not in tags:
        raise Exception(f"{image} does not have a {LATEST_TAG} tag")

    # Sort tags in descending order because this should be faster,
    # and move latest to the beginning to ensure we get its digest first
    tags_sorted = sorted([t for t in tags if t != LATEST_TAG], reverse=True)
    tags_sorted.insert(0, LATEST_TAG)

    # Find another tag corresponding to latest
    latest_digest = None
    latest_tag = None
    for tag in tags_sorted:
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
        logger.log_debug(f"Finding latest image tag/digest for {repo}")
        suffix = get_latest_image_suffix(repo)
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
        print(contents)
    else:
        if output_file is None:
            output_file = input_file
        with open(output_file, "w") as f:
            f.write(contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, help="File containing images to pin to latest versions", required=True)
    parser.add_argument("-o", "--output", type=Path, help="File to which output will be written. Defaults to the input file if not specified.")
    args = parser.parse_args()

    output = args.output or args.input
    transform_file(args.input, output)
