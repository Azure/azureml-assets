# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import re
from pathlib import Path
from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.models.search_scope import SearchScope
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.network.session import PipSession
from typing import List

from azureml.assets.util import logger

LATEST_PYPI_VERSION = re.compile(r"([^\"'\s]+)([=~]=)\{\{latest-pypi-version\}\}")
PYPI_URL = "https://pypi.org/simple"


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
    try:
        return PackageFinder.create(
            link_collector=link_collector,
            selection_prefs=selection_prefs,
        )
    except TypeError:
        # Handle pip>=22.0
        return PackageFinder.create(
            link_collector=link_collector,
            selection_prefs=selection_prefs,
            use_deprecated_html5lib=False
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


def pin_packages(contents: str) -> str:
    # Process pip template tags
    package_finder = create_package_finder([PYPI_URL])
    while True:
        match = LATEST_PYPI_VERSION.search(contents)
        if not match:
            break
        package = match.group(1)
        selector = match.group(2)
        logger.log_debug(f"Looking up latest version of {package}")
        version = get_latest_package_version(package, package_finder)
        logger.log_debug(f"Latest version of {package} is {version}")
        contents = contents[:match.start()] + f"{package}{selector}{version}" + contents[match.end():]

    return contents


def transform_file(input_file: Path, output_file: Path = None):
    # Read file
    with open(input_file) as f:
        contents = f.read()

    # Transform
    contents = pin_packages(contents)

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
    parser.add_argument("-i", "--input", type=Path, help="File containing packages to pin to latest versions", required=True)
    parser.add_argument("-o", "--output", type=Path, help="File to which output will be written. Defaults to the input file if not specified.")
    args = parser.parse_args()

    output = args.output or args.input
    transform_file(args.input, output)
