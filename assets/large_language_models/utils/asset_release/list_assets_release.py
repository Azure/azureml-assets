# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for listing recent AzureML asset releases from Azure DevOps.

This script queries Azure DevOps pipelines to retrieve the most recent
AzureML Asset release runs and their associated build information, with
optional filtering by a provided build pattern.

Environment Variables
---------------------
AZURE_DEVOPS_PAT : Personal Access Token used for authenticating to
    Azure DevOps (required).
ASSET_RELEASE_COUNT : Default number of recent releases to display when
    the ``--number`` argument is not supplied (optional).

Typical usage (from repo root):
    python assets/large_language_models/utils/asset_release/ \
        list_assets_release.py --number 5 --pattern nightly --verbose
"""

import argparse
import os

from dotenv import load_dotenv
from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication

load_dotenv()

# Retrieve the PAT from the environment variable
personal_access_token = os.getenv("AZURE_DEVOPS_PAT")
if personal_access_token is None:
    print("Please set AZURE_DEVOPS_PAT in your environment variable")
    exit()

# Azure DevOps org URL and Asset Release/Build definition ID
organization_url = "https://dev.azure.com/msdata"
Project_Name = "Vienna"

AzureML_Assets_Release_Definition_ID = 37560
AzureML_Assets_Build_Definition_ID = 37266
DEFAULT_LAST_RELEASES = 10

if not AzureML_Assets_Release_Definition_ID:
    AzureML_Assets_Release_Definition_ID = int(
        input("Enter Release Pipeline Definition ID: ")
    )

if not AzureML_Assets_Build_Definition_ID:
    AzureML_Assets_Build_Definition_ID = int(
        input("Enter Build Pipeline Definition ID: ")
    )

# Allow overriding default via environment variable
DEFAULT_LAST_RELEASES = int(
    os.getenv("ASSET_RELEASE_COUNT", DEFAULT_LAST_RELEASES)
)

# Create a connection to the org
credentials = BasicAuthentication("", personal_access_token)
connection = Connection(base_url=organization_url, creds=credentials)


def get_last_n_releases(args):
    """Return metadata for the last N asset releases.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing:
          * number (int): maximum number of releases to return.
          * pattern (str|None): optional substring filter applied to the
            build pattern captured at queue time.
          * verbose (bool): whether verbose output was requested (only
            impacts printing in ``main``; not used here for logic).

    Returns
    -------
    list[dict]
        A list (up to ``args.number`` elements) of dictionaries with keys:
        ``release_name``, ``release_status``, ``release_version``,
        ``release_url``, ``build_version``, ``release_description``,
        ``build_name``, ``build_url``, and ``build_pattern``.
    """
    asset_releases = []

    # Get a release client
    release_client = connection.clients.get_pipelines_client()
    # NOTE: could filter by created time; left for future enhancement

    # Get a build client
    build_client = connection.clients.get_pipelines_client()

    while True:
        print("Please Wait...", flush=True)

        releases = release_client.list_runs(
            project=Project_Name,
            pipeline_id=AzureML_Assets_Release_Definition_ID,
        )

        print(
            "Pipeline Name: ",
            release_client.get_pipeline(
                project=Project_Name,
                pipeline_id=AzureML_Assets_Release_Definition_ID,
            ).name,
        )

        builds = build_client.list_runs(
            project=Project_Name,
            pipeline_id=AzureML_Assets_Build_Definition_ID,
        )

        if releases:
            for release in releases:
                rls_item = {}

                rls_item["release_name"] = release.name
                rls_item["release_status"] = release.result
                rls_item["release_version"] = release.id
                rls_item["release_url"] = release.url
                template_params = release.additional_properties.get(
                    "templateParameters", {}
                )
                rls_item["build_version"] = template_params.get(
                    "sourceVersion"
                )
                rls_item["release_description"] = template_params.get(
                    "description"
                )

                build = next(
                    (
                        b
                        for b in builds
                        if rls_item["build_version"]
                        and rls_item["build_version"] in b.name
                    ),
                    None,
                )
                rls_item["build_name"] = (
                    build.name if build else "Unknown Build"
                )
                rls_item["build_url"] = (
                    build.url if build else "Unknown Build URL"
                )

                try:
                    queue_time_variables = build.variables
                    rls_item["build_pattern"] = queue_time_variables[
                        "pattern"
                    ].value
                except Exception:
                    rls_item["build_pattern"] = None

                if rls_item["build_pattern"]:
                    if (
                        args.pattern is None
                        or args.pattern.lower()
                        in rls_item["build_pattern"].lower()
                    ):
                        asset_releases.append(rls_item)
                if len(asset_releases) >= args.number:
                    print("")
                    return asset_releases[: args.number]
        else:
            break
    print("")
    return asset_releases[: args.number]


def main():
    """Parse command-line arguments and print recent asset releases.

    This is the script entry point. It gathers CLI options, invokes
    :func:`get_last_n_releases`, and prints a human-readable summary of each
    retrieved release, respecting the ``--verbose`` flag.
    """
    # Initialize parser
    parser = argparse.ArgumentParser(
        description=(
            "Quick script to list recent AzureML Asset releases."
        )
    )

    # Adding optional argument
    parser.add_argument(
        "-v", "--verbose", help="Show verbose output", action="store_true"
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=DEFAULT_LAST_RELEASES,
        help=(
            f"Last n releases, default is {DEFAULT_LAST_RELEASES}"
        ),
    )

    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        help="Filter by build pattern, default is None.",
    )

    # Read arguments from command line
    args = parser.parse_args()

    if args.verbose:
        print("Verbose output is turned on")

    if args.number:
        message = f"Searching for the last {args.number} asset releases"
        if args.pattern:
            message += f" with build pattern '{args.pattern}':"
        print(message)

    asset_releases = get_last_n_releases(args)

    for rls in asset_releases:
        print(f"Release Name: {rls['release_name']}")
        if args.verbose:
            print(f"Release Description: {rls['release_description']}")
            print(f"Release Status: {rls['release_status']}")
            print(f"Release URL: {rls['release_url']}")
        print(f"Build Pattern: {rls['build_pattern']}")
        print(f"Build Name: {rls['build_name']}")
        print(f"Build URL: {rls['build_url']}")
        print("-" * 30)


if __name__ == "__main__":
    main()
