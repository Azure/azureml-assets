import argparse
import json
import os
import requests
from urllib.parse import urljoin

from ci_logger import logger

ENV_URL_TEMPLATE = "environment/1.0/environments/{environment}/versions/{version}/registry/{registry}"


def create_environment(base_url: str,
                       name: str,
                       version: str,
                       registry: str,
                       image_name: str,
                       access_token: str,
                       env_def_with_metadata: object):
    url = urljoin(base_url, ENV_URL_TEMPLATE.format(environment=name, version=version, registry=registry))
    headers = {
        'Authorization': f"Bearer {access_token}",
        'Content-Type': "application/json"
    }
    params = {'imageName': image_name}
    response = requests.put(url, headers=headers, json=env_def_with_metadata, params=params)
    if not response.ok:
        logger.log_error(f"Failed to create {name} version {version}: {response.status_code} {response.text}")
    return response.ok


def create_environments(deployment_config_file_path: str,
                        base_url: str,
                        registry: str,
                        access_token: str,
                        version_template: str = None):
    # Load config
    path = os.path.dirname(os.path.abspath(deployment_config_file_path))
    with open(deployment_config_file_path) as f:
        deployment_config = json.loads(f.read())

    # Iterate over environments
    errors = False
    for name, values in deployment_config.items():
        print(f"Creating environment {name} in {registry}")

        # Coerce values into a list
        values_list = values if isinstance(values, list) else [values]

        # Iterate over versions, although there's likely just one
        for value in values_list:
            version = value['version']

            # Load EnvironmentDefinitionWithSetMetadataDto
            with open(os.path.join(path, value['path'])) as f:
                env_def_with_metadata = json.loads(f.read())

            # Get published full image name
            full_image_name = value['publish']['fullImageName']

            # Apply version template
            updated_version = version_template.format(version=version) if version_template else version

            # Create environment
            success = create_environment(base_url=base_url, name=name, version=updated_version, registry=registry,
                                         image_name=full_image_name, access_token=access_token,
                                         env_def_with_metadata=env_def_with_metadata)
            if not success:
                errors = True

    # Final messages
    if errors:
        raise Exception("Errors occurred while creating environments") 


if __name__ == "__main__":
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--deployment-config", required=True, help="Path to deployment config file")
    parser.add_argument("-u", "--url", required=True, help="Base AzureML URL, example: https://master.api.azureml-test.ms")
    parser.add_argument("-r", "--registry", required=True, help="Name of the target registry")
    parser.add_argument("-t", "--token", required=True, help="Access token to use for bearer authentication")
    parser.add_argument("-v", "--version-template", help="Template to apply to the version in the deployment config "
                        "file, example: '{version}.dev1'")
    args = parser.parse_args()

    create_environments(deployment_config_file_path=args.deployment_config,
                        base_url=args.url,
                        registry=args.registry,
                        access_token=args.token,
                        version_template=args.version_template)
