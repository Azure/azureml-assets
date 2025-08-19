import argparse
import json
import os

from dotenv import load_dotenv
load_dotenv()

from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication

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
Last_N_Releases = None

if not AzureML_Assets_Release_Definition_ID:
    AzureML_Assets_Release_Definition_ID = int(input("Enter Release Pipeline Definition ID: "))

if not AzureML_Assets_Build_Definition_ID:
    AzureML_Assets_Build_Definition_ID = int(input("Enter Build Pipeline Definition ID: "))

if not Last_N_Releases:
    Last_N_Releases = int(input("Enter the number of last releases to retrieve: "))

# Create a connection to the org
credentials = BasicAuthentication("", personal_access_token)
connection = Connection(base_url=organization_url, creds=credentials)


def get_last_n_releases(args):
    asset_releases = []

    # Get a release client
    release_client = connection.clients.get_pipelines_client()
    max_created_time = None

    # Get a build client
    build_client = connection.clients.get_pipelines_client()

    while True:
        print("Please Wait...", flush=True)

        releases = release_client.list_runs(
            project=Project_Name,
            pipeline_id=AzureML_Assets_Release_Definition_ID,
        )

        print('Pipeline Name: ', release_client.get_pipeline(project=Project_Name, pipeline_id=AzureML_Assets_Release_Definition_ID).name)

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
                rls_item["build_version"] = release.additional_properties.get("templateParameters").get("sourceVersion")
                rls_item["release_description"] = release.additional_properties.get("templateParameters").get("description")

                build = next((b for b in builds if rls_item["build_version"] in b.name), None)  
                rls_item["build_name"] = build.name if build else "Unknown Build"
                rls_item["build_url"] = build.url if build else "Unknown Build URL"

                try:
                    queue_time_variables = build.variables
                    rls_item["build_pattern"] = queue_time_variables["pattern"].value
                except Exception as e:
                    rls_item["build_pattern"] = None

                if rls_item["build_pattern"]:
                    if (
                        args.pattern is None
                        or args.pattern.lower() in rls_item["build_pattern"].lower()
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
    global Last_N_Releases
    # Initialize parser
    parser = argparse.ArgumentParser(
        description="This is quick script to get the latest release of AzureML Assets."
    )

    # Adding optional argument
    parser.add_argument(
        "-v", "--verbose", help="Show verbose output", action="store_true"
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=Last_N_Releases,
        help=f"Last n releases, default is {Last_N_Releases}",
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
        print("-"*30)


if __name__ == "__main__":
    main()
