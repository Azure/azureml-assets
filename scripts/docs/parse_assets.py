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
                 pattern: re.Pattern = None):
    """Parse all assets from input directory and generate documentation for each.

    Args:
        input_dirs (List[Path]): List of directories to search for assets.
        asset_config_filename (str): Asset config filename to search for.
        pattern (re.Pattern, optional): Regex pattern for assets to parse. Defaults to None.
    """
    asset_count = 0

    references = {}

    for asset_config in util.find_assets(input_dirs, asset_config_filename, pattern=pattern):
        asset_count += 1

        asset_type, asset_name, asset_file_name, asset_description = \
            generate_asset_documentation.create_asset_doc(asset_config, asset_config.type)

        if asset_type not in references:
            references[asset_type] = defaultdict(list)

        for category in asset_config.categories:
            references[asset_type][category].append((asset_name, asset_file_name, asset_description))
        else:
            # Put the asset under "Uncategorized"
            references[asset_type]["Uncategorized"].append((asset_name, asset_file_name, asset_description))

        references[asset_type]["All"].append((asset_name, asset_file_name, asset_description))

    logger.print(f"{asset_count} asset(s) parsed")

    for asset_type in references:
        category_docs_links_list = []

        for category in references[asset_type]:
            if category == "All":
                continue

            category_doc_file_name = f"{asset_type}s-{category.replace(' ', '-')}-documentation"
            capitalized_category = category[0].upper() + category[1:]

            # Add to unordered list of categories
            category_docs_links_list.append(
                snakemd.Paragraph(capitalized_category).insert_link(capitalized_category, category_doc_file_name))

            # Create a new markdown file for each category
            category_doc = snakemd.new_doc()
            category_doc.add_heading(f"{capitalized_category} {asset_type}s", level=1)

            # Create list of assets under the category
            category_asset_links_list = []

            for asset_name, asset_file_name, asset_description in references[asset_type][category]:
                category_asset_links_list.append(
                    snakemd.Paragraph(asset_name).insert_link(asset_name, category_doc_file_name))

            category_doc.add_unordered_list(category_asset_links_list)

            # Write to category doc
            with open(f"{asset_type}s/{category_doc_file_name}.md", 'w') as f:
                f.write(str(category_doc))

        # Create a new markdown file for each asset type
        doc = snakemd.new_doc()
        doc.add_heading(f"{asset_type.capitalize()}s", level=1)

        # Add list of asset categories
        doc.add_heading("By category", level=2)
        doc.add_unordered_list(category_docs_links_list)

        # alphabetize references with case-insensitivity
        references[asset_type]["All"].sort(key=lambda x: x[0].lower())

        # Create glossary that links to each asset of the asset type
        doc.add_heading("Glossary", level=2)

        doc.add_horizontal_rule()

        asset_links_list = []
        for asset_name, asset_file_name, asset_description in references[asset_type]["All"]:
            doc.add_unordered_list([snakemd.Paragraph(asset_name).insert_link(asset_name, asset_file_name)])
            # limit description to 300 chars
            description = asset_description if len(asset_description) <= 300 else (asset_description[:298] + "...")
            doc.add_raw("\n  > " + description)

        doc.add_unordered_list(asset_links_list)

        with open(f"{asset_type}s/{asset_type}s-documentation.md", 'w') as f:
            f.write(str(doc))


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True,
                        help="Comma-separated list of directories containing assets")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME,
                        help="Asset config file name to search for")
    parser.add_argument("-t", "--pattern", type=re.compile,
                        help="Regex pattern to select assets to copy, in the format <type>/<name>/<version>")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]

    # Parse assets
    parse_assets(input_dirs=input_dirs,
                 asset_config_filename=args.asset_config_filename,
                 pattern=args.pattern)
