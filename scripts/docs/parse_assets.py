# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Parse assets from directory and generate markdown files."""
import generate_asset_documentation

import argparse
import re
from pathlib import Path
from typing import List

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger

import snakemd
from collections import defaultdict

PARSED_COUNT = "parsed_count"



def parse_assets(input_dirs: List[Path],
                asset_config_filename: str,
                release_directory_root: Path = None,
                use_version_dirs: bool = False,
                pattern: re.Pattern = None):
    """Parse all assets from input directory and generate documentation for each.

    Args:
        input_dirs (List[Path]): List of directories to search for assets.
        asset_config_filename (str): Asset config filename to search for.
        release_directory_root (Path, optional): Release directory location. Defaults to None.
        use_version_dirs (bool, optional): Use version directories for output. Defaults to False.
        pattern (re.Pattern, optional): Regex pattern for assets to copy. Defaults to None.
    """
    # Find assets under release dir
    asset_count = 0

    references = defaultdict(list)

    for asset_config in util.find_assets(input_dirs, asset_config_filename, pattern=pattern):
        asset_count += 1

        asset_type, asset_file_name = generate_asset_documentation.create_asset_doc(asset_config.spec_with_path, asset_config.type)
        references[asset_type].append(asset_file_name)

    logger.print(f"{asset_count} asset(s) copied")


    for asset_type in references:


        print(asset_type)
        # create new md file (folder_name + ".md")

        doc = snakemd.new_doc()
        doc.add_heading(asset_type.capitalize() + "s", level=1)

        doc.add_heading("Glossary", level=2)

        # [componentA]: https://github.com/vizhur/vizhur/wiki/componentA
        # [componentB]: https://github.com/vizhur/vizhur/wiki/componentB
        # [componentC]: https://github.com/vizhur/vizhur/wiki/componentC

        # ***
        # * [ComponentA][componentA]

        for asset_file_name in references[asset_type]:
            doc.add_paragraph("[" + asset_file_name + "]: https://github.com/vizhur/vizhur/wiki/" + asset_type + "-" + asset_file_name)
            
        doc.add_horizontal_rule()  # aka ***

        asset_links_list = []
        for asset_file_name in references[asset_type]:
            asset_links_list.append(snakemd.Paragraph(asset_file_name).insert_link(asset_file_name, asset_file_name))
            # add xref link to Glossary!

        doc.add_unordered_list(asset_links_list)


        with open(asset_type + "s/" + asset_type + "s.md", 'w') as f:
            f.write(str(doc))




if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True,
                        help="Comma-separated list of directories containing assets")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME,
                        help="Asset config file name to search for")
    parser.add_argument("-r", "--release-directory", type=Path,
                        help="Directory to which the release branch has been cloned. "
                        "If provided, only asset versions without a release tag will be copied.")
    parser.add_argument("-v", "--use-version-dirs", action="store_true",
                        help="Use version directories when storing assets in output directory")
    parser.add_argument("-t", "--pattern", type=re.compile,
                        help="Regex pattern to select assets to copy, in the format <type>/<name>/<version>")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]

    # Copy assets
    parse_assets(input_dirs=input_dirs,
                asset_config_filename=args.asset_config_filename,
                release_directory_root=args.release_directory,
                use_version_dirs=args.use_version_dirs,
                pattern=args.pattern)
