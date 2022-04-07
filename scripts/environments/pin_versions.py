import argparse
import json
import re
import urllib.parse
from ci_logger import logger
from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.models.search_scope import SearchScope
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.network.session import PipSession
from typing import List
from urllib.request import Request, urlopen    

LATEST_TAG = "latest"
LATEST_IMAGE_TAG = re.compile(r"([^\"'\s]+):\{\{latest-image-tag\}\}")
LATEST_PYPI_VERSION = re.compile(r"([^\"'\s]+)([=~]=)\{\{latest-pypi-version\}\}")
PYPI_URL = "https://pypi.org/simple"


def get_latest_image_suffix(image: str):
    (hostname, repo) = image.split("/", 1)

    # Retrieve tags
    response = urlopen(f"https://{hostname}/v2/{repo}/tags/list")
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
        response = urlopen(request)
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


def create_package_finder(index_urls: List[str]) -> PackageFinder:
    """
    Create a pip PackageFinder.
    """
    link_collector = LinkCollector(
        session=PipSession(),
        search_scope=SearchScope([], index_urls),
    )
    selection_prefs = SelectionPreferences(
        allow_yanked=True,
    )
    return PackageFinder.create(
        link_collector=link_collector,
        selection_prefs=selection_prefs,
    )


def get_latest_package_version(package: str,
                               package_finder: PackageFinder,
                               include_pre: bool = False) -> str:
    for _ in range(5):
        try:
            candidates = package_finder.find_all_candidates(package)
            versions = []
            for c in candidates:
                versions.append(c.version)
            if len(versions) > 0:
                versions.sort(reverse=True)
                if include_pre:
                    return str(versions[0])
                else:
                    for v in versions:
                        if not v.is_prerelease:
                            return str(v)
        except Exception as e:
            logger.log_warning(f"Failed to find candidates for {package}: {e}")
            continue
    return None


def transform(input_file: str, output_file: str = None):
    # Output to input file by default
    if output_file is None:
        output_file = input_file

    # Read Dockerfile
    with open(input_file) as f:
        contents = f.read()

    # Process MCR template tags
    while True:
        match = LATEST_IMAGE_TAG.search(contents)
        if not match:
            break
        repo = match.group(1)
        print(f"Finding latest image tag/digest for {repo}")
        suffix = get_latest_image_suffix(repo)
        print(f"Latest image reference is {repo}{suffix}")
        contents = contents[:match.start()] + f"{repo}{suffix}" + contents[match.end():]

    # Process pip template tags
    package_finder = create_package_finder([PYPI_URL])
    while True:
        match = LATEST_PYPI_VERSION.search(contents)
        if not match:
            break
        package = match.group(1)
        selector = match.group(2)
        print(f"Looking up latest version of {package}")
        version = get_latest_package_version(package, package_finder)
        print(f"Latest version of {package} is {version}")
        contents = contents[:match.start()] + f"{package}{selector}{version}" + contents[match.end():]

    # Write to stdout or output_file
    if output_file == "-":
        print(contents)
    else:
        with open(output_file, "w") as f:
            f.write(contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="File containing images/packages to pin to latest versions", required=True)
    parser.add_argument("-o", "--output", help="File to which output will be written. Defaults to the input file if not specified.")
    args = parser.parse_args()

    output = args.output or args.input
    transform(args.input, output)