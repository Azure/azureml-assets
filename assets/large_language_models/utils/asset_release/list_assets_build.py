# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
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
AzureML_Assets_Build_Definition_ID = 37266
Main_Branch_Name = "refs/heads/main"

Last_N_Build = 3

# Create a connection to the org
credentials = BasicAuthentication("", personal_access_token)
connection = Connection(base_url=organization_url, creds=credentials)


def get_last_n_build(args):
    asset_builds = []
    build_client = connection.clients.get_build_client()
    builds = build_client.get_builds(
        project=Project_Name,
        definitions=[AzureML_Assets_Build_Definition_ID],
        branch_name=Main_Branch_Name,
    )
    for build in builds:
        build_item = {}
        build_item["build_number"] = build.build_number
        build_item["build_id"] = build.id
        build_item["build_created_by"] = build.requested_by.display_name
        build_item["build_requested_by"] = build.requested_by.unique_name
        build_parameters = json.loads(build.parameters)
        build_item["build_pattern"] = build_parameters.get("pattern", "N/A")
        build_item["build_time"] = build.finish_time
        build_item["build_result"] = build.result
        if args.succeeded and build.result != "succeeded":
            continue
        build_item["build_status"] = build.status
        build_item["build_url"] = (
            "https://dev.azure.com/msdata/Vienna/_build/results?buildId="
            f"{build.id}&view=results"
        )

        if (
            args.pattern is None
            or args.pattern.lower() in build_item["build_pattern"].lower()
        ):
            asset_builds.append(build_item)

    return asset_builds[:args.number]


def main():
    # Initialize parser
    parser = argparse.ArgumentParser(
        description="Quick script to list recent AzureML Asset builds."
    )

    # Adding optional argument
    parser.add_argument(
        "-v", "--verbose", help="Show verbose output", action="store_true"
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=Last_N_Build,
        help=f"Last n releases, default is {Last_N_Build}",
    )

    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        help="Filter by build pattern, default is None.",
    )

    parser.add_argument(
        "-s",
        "--succeeded",
        help="Show only succeeded builds",
        action="store_true",
        default=True,
    )
    # Read arguments from command line
    args = parser.parse_args()

    if args.verbose:
        print("Verbose output is turned on")

    if args.number:
        message = f"Searching for the last {args.number} asset builds."
        if args.pattern:
            message += f" with build pattern '{args.pattern}':"
        print(message)

    asset_builds = get_last_n_build(args)

    for build in asset_builds:
        print(f"Build_number: {build['build_number']}")
        print(f"Build Pattern: {build['build_pattern']}")
        print(f"Build Status: {build['build_status']}")
        print(f"Build Result: {build['build_result']}")
        print(
            "Create Time: "
            f"{build['build_time'].strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(
            f"Created By: {build['build_created_by']} "
            f"({build['build_requested_by']})"
        )
    print(f"Build URL: {build['build_url']}")
    # TODO: if verbose, also print environment version (future enhancement)
    print("")


if __name__ == "__main__":
    main()
