# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLIndex auth connection utilities."""
import json
import os
import re
from typing import Optional, Union
from distutils.version import LooseVersion
from contextlib import suppress

import pkg_resources

from logging_utilities import get_logger

try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import WorkspaceConnection
except Exception:
    MLClient = None
    WorkspaceConnection = None
try:
    from azure.core.credentials import TokenCredential
except Exception:
    TokenCredential = object

Connection = None
logger = get_logger(name="connections")

pkg_versions = {
    "azure-ai-ml": ""
}

AZURE_AI_ML_MIN_VERSION = LooseVersion("1.10.0")

for package in pkg_versions:
    with suppress(Exception):
        pkg_versions[package] = LooseVersion(pkg_resources.get_distribution(package).version)


def create_session_with_retry(retry=3):
    """
    Create requests.session with retry.

    :type retry: int
    rtype: Response
    """
    import requests
    from requests.adapters import HTTPAdapter

    retry_policy = _get_retry_policy(num_retry=retry)

    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry_policy))
    session.mount("http://", HTTPAdapter(max_retries=retry_policy))
    return session


def _get_retry_policy(num_retry=3):
    """
    Request retry policy with increasing backoff.

    :return: Returns the msrest or requests REST client retry policy.
    :rtype: urllib3.Retry
    """
    from urllib3 import Retry

    status_forcelist = [413, 429, 500, 502, 503, 504]
    backoff_factor = 0.4
    retry_policy = Retry(
        total=num_retry,
        read=num_retry,
        connect=num_retry,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        # By default this is True. We set it to false to get the full error trace, including url and
        # status code of the last retry. Otherwise, the error message is 'too many 500 error responses',
        # which is not useful.
        raise_on_status=False
    )
    return retry_policy


def send_post_request(url, headers, payload):
    """Send a POST request."""
    with create_session_with_retry() as session:
        response = session.post(url, data=json.dumps(payload), headers=headers)
        # Raise an exception if the response contains an HTTP error status code
        response.raise_for_status()

    return response


def get_connection_credential(config, credential: Optional[TokenCredential] = None):
    """Get a credential for a connection."""
    try:
        from azure.core.credentials import AzureKeyCredential
    except ImportError as e:
        raise ValueError(
            "Could not import azure-core python package. "
            "Please install it with `pip install azure-core`."
        ) from e
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError as e:
        raise ValueError(
            "Could not import azure-identity python package. "
            "Please install it with `pip install azure-identity`."
        ) from e

    if config.get("connection_type", None) == "workspace_keyvault":
        from azureml.core import Run, Workspace
        run = Run.get_context()
        if hasattr(run, "experiment"):
            ws = run.experiment.workspace
        else:
            try:
                ws = Workspace(
                    subscription_id=config.get("connection", {}).get("subscription"),
                    resource_group=config.get("connection", {}).get("resource_group"),
                    workspace_name=config.get("connection", {}).get("workspace")
                )
            except Exception as e:
                logger.warning(f"Could not get workspace '{config.get('connection', {}).get('workspace')}': {e}")
                # Fall back to looking for key in environment.
                import os
                key = os.environ.get(config.get("connection", {}).get("key"))
                if key is None:
                    raise ValueError(f"Could not get workspace \
                                     '{config.get('connection', {}).get('workspace')}' and no key named \
                                     '{config.get('connection', {}).get('key')}' in environment")
                return AzureKeyCredential(key)

        keyvault = ws.get_default_keyvault()
        connection_credential = AzureKeyCredential(keyvault.get_secret(config.get("connection", {}).get("key")))
    elif config.get("connection_type", None) == "workspace_connection":
        connection_id = config.get("connection", {}).get("id")
        connection = get_connection_by_id_v2(connection_id, credential=credential)
        connection_credential = connection_to_credential(connection)
    elif config.get("connection_type", None) == "environment":
        import os
        key = os.environ.get(config.get("connection", {}).get("key", "OPENAI_API_KEY"))
        if key is None:
            if credential is not None:
                connection_credential = credential
            else:
                connection_credential = DefaultAzureCredential(process_timeout=60)
        else:
            connection_credential = AzureKeyCredential(key)
    else:
        connection_credential = credential if credential is not None else DefaultAzureCredential(process_timeout=60)

    return connection_credential


def workspace_connection_to_credential(connection: Union[dict, Connection, WorkspaceConnection]):
    """Get a credential for a workspace connection."""
    return connection_to_credential(connection)


def connection_to_credential(connection: Union[dict, Connection, WorkspaceConnection]):
    """Get a credential for a workspace connection."""
    if isinstance(connection, dict):
        props = connection["properties"]
        auth_type = props.get("authType", props.get("AuthType"))
        if auth_type == "ApiKey":
            from azure.core.credentials import AzureKeyCredential
            return AzureKeyCredential(props["credentials"]["key"])
        elif auth_type == "PAT":
            from azure.core.credentials import AccessToken
            return AccessToken(props["credentials"]["pat"], props.get("expiresOn", None))
        elif auth_type == "CustomKeys":
            # OpenAI connections are made with CustomKeys auth, so we can try to access the key using known structure
            from azure.core.credentials import AzureKeyCredential
            if connection.get("metadata", {}).get(
                    "azureml.flow.connection_type",
                    connection.get('ApiType', connection.get('apiType', ''))).lower() == "openai":
                # Try to get the the key with api_key, if fail, default to regular CustomKeys handling
                try:
                    key = props["credentials"]["keys"]["api_key"]
                    return AzureKeyCredential(key)
                except Exception as e:
                    logger.warning(f"Could not get key using api_key, using default handling: {e}")
            key_dict = props["credentials"]["keys"]
            if len(key_dict.keys()) != 1:
                raise ValueError(f"Only connections with a single key can be used. \
                                 Number of keys present: {len(key_dict.keys())}")
            return AzureKeyCredential(props["credentials"]["keys"][list(key_dict.keys())[0]])
        else:
            raise ValueError(f"Unknown auth type '{auth_type}'")
    elif isinstance(connection, WorkspaceConnection):
        if connection.credentials.type.lower() == "api_key":
            from azure.core.credentials import AzureKeyCredential
            return AzureKeyCredential(connection.credentials.key)
        elif connection.credentials.type.lower() == "pat":
            from azure.core.credentials import AccessToken
            return AccessToken(connection.credentials.pat, connection.credentials.expires_on)
        elif connection.credentials.type.lower() == "custom_keys":
            if connection._metadata.get("azureml.flow.connection_type", "").lower() == "openai":
                from azure.core.credentials import AzureKeyCredential
                try:
                    key = connection.credentials.keys.api_key
                    return AzureKeyCredential(key)
                except Exception as e:
                    logger.warning(f"Could not get key using api_key, using default handling: {e}")
            key_dict = connection.credentials.keys
            if len(key_dict.keys()) != 1:
                raise ValueError(f"Only connections with a single key can be used. \
                                 Number of keys present: {len(key_dict.keys())}")
            return AzureKeyCredential(connection.credentials.keys[list(key_dict.keys())[0]])
        else:
            raise ValueError(f"Unknown auth type '{connection.credentials.type}' for connection '{connection.name}'")
    else:
        if connection.credentials.type.lower() == "api_key":
            from azure.core.credentials import AzureKeyCredential
            return AzureKeyCredential(connection.credentials.key)
        else:
            raise ValueError(f"Unknown auth type '{connection.credentials.type}' for connection '{connection.name}'")


def snake_case_to_camel_case(s):
    """Convert snake case to camel case."""
    first = True
    final = ""
    for word in s.split("_"):
        if first:
            first = False
            final += word
        else:
            final += word.title()
    return final


def recursive_dict_keys_snake_to_camel(d: dict, skip_keys=[]) -> dict:
    """Convert snake case to camel case in dict keys."""
    new_dict = {}
    for k, v in d.items():
        if k not in skip_keys:
            if isinstance(v, dict):
                v = recursive_dict_keys_snake_to_camel(v, skip_keys=skip_keys)
            if isinstance(k, str):
                new_key = snake_case_to_camel_case(k)
                new_dict[new_key] = v
        else:
            new_dict[k] = v
    return new_dict


def get_connection_by_id_v2(
        connection_id: str,
        credential: TokenCredential = None,
        client: str = "sdk") -> Union[dict, WorkspaceConnection, Connection]:
    """
    Get a connection by id using azure.ai.ml or azure.ai.generative.

    If azure.ai.ml is installed, use that, otherwise use azure.ai.generative.
    """
    uri_match = re.match(
        r"/subscriptions/(.*)/resourceGroups/(.*)/providers/Microsoft.MachineLearningServices" +
        r"/workspaces/(.*)/connections/(.*)",
        connection_id, flags=re.IGNORECASE
    )

    if uri_match is None:
        logger.warning(f"Invalid connection_id {connection_id}, expecting Azure Machine Learning resource ID.\
                       Trying with current workspace.")
        from run_utils import TestRun
        run = TestRun()
        subscriptio_id = run.subscription
        ws_name = run.workspace.name
        resource_group = run.workspace.resource_group
        connection_string = connection_id

    else:
        subscriptio_id = uri_match.group(1)
        resource_group = uri_match.group(2)
        ws_name = uri_match.group(3)
        connection_string = uri_match.group(4)
    logger.info(f"Getting workspace connection: {connection_string}")

    from azureml.dataprep.api._aml_auth._azureml_token_authentication import AzureMLTokenAuthentication

    if credential is None:
        from azure.identity import DefaultAzureCredential

        if os.environ.get("AZUREML_RUN_ID", None) is not None:
            credential = AzureMLTokenAuthentication._initialize_aml_token_auth()
        else:
            credential = DefaultAzureCredential(process_timeout=60)

    logger.info(f"Using auth: {type(credential)}")

    if client == "sdk" and MLClient is not None and pkg_versions["azure-ai-ml"] >= AZURE_AI_ML_MIN_VERSION:
        logger.info("Getting workspace connection via MLClient")
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscriptio_id,
            resource_group_name=resource_group,
            workspace_name=ws_name
        )

        if os.environ.get("AZUREML_RUN_ID", None) is not None:
            # In AzureML Run context, we need to use workspaces internal endpoint that will accept AzureMLToken auth.
            old_base_url = ml_client.connections._operation._client._base_url
            new_base_url = f"{os.environ.get('AZUREML_SERVICE_ENDPOINT')}/rp/workspaces"
            ml_client.connections._operation._client._base_url = new_base_url

        logger.info(f"Using ml_client base_url: {ml_client.connections._operation._client._base_url}")

        list_secrets_response = ml_client.connections._operation.list_secrets(
            connection_name=connection_string,
            resource_group_name=ml_client.resource_group_name,
            workspace_name=ml_client.workspace_name,
        )
        try:
            connection = WorkspaceConnection._from_rest_object(list_secrets_response)
            logger.info(f"Parsed Connection: {connection.id}")
        except Exception as e:
            logger.warning(f"Failed to parse connection into azure-ai-ml sdk object, returning response as is: {e}")
            connection = recursive_dict_keys_snake_to_camel(list_secrets_response.as_dict(),
                                                            skip_keys=["credentials", "metadata"])

        if os.environ.get("AZUREML_RUN_ID", None) is not None:
            ml_client.connections._operation._client._base_url = old_base_url
    else:
        logger.info("Getting workspace connection via REST as fallback")
        return get_connection_by_id_v1(connection_id, credential)

    return connection


def get_id_from_connection(connection: Union[dict, WorkspaceConnection, Connection]) -> str:
    """Get a connection id from a connection."""
    if isinstance(connection, dict):
        return connection["id"]
    elif isinstance(connection, WorkspaceConnection):
        return connection.id
    elif isinstance(connection, Connection):
        return connection.id
    else:
        raise ValueError(f"Unknown connection type: {type(connection)}")


def get_target_from_connection(connection: Union[dict, WorkspaceConnection, Connection]) -> str:
    """Get a connection target from a connection."""
    if isinstance(connection, dict):
        return connection["properties"]["target"]
    elif isinstance(connection, WorkspaceConnection):
        return connection.target
    elif isinstance(connection, Connection):
        return connection.target
    else:
        raise ValueError(f"Unknown connection type: {type(connection)}")


def get_metadata_from_connection(connection: Union[dict, WorkspaceConnection, Connection]) -> dict:
    """Get a connection metadata from a connection."""
    if isinstance(connection, dict):
        return connection["properties"]["metadata"]
    elif isinstance(connection, WorkspaceConnection):
        return connection.metadata
    elif isinstance(connection, Connection):
        return connection.metadata
    else:
        raise ValueError(f"Unknown connection type: {type(connection)}")


def get_connection_by_name_v2(workspace, name: str) -> dict:
    """Get a connection from a workspace."""
    if hasattr(workspace._auth, "get_token"):
        bearer_token = workspace._auth.get_token("https://management.azure.com/.default").token
    else:
        bearer_token = workspace._auth.token

    endpoint = workspace.service_context._get_endpoint("api")
    url = f"{endpoint}/rp/workspaces/subscriptions/{workspace.subscription_id}\
            /resourcegroups/{workspace.resource_group}\
            /providers/Microsoft.MachineLearningServices/workspaces/{workspace.name}\
            /connections/{name}/listsecrets?api-version=2023-02-01-preview"
    resp = send_post_request(url, {
        "Authorization": f"Bearer {bearer_token}",
        "content-type": "application/json"
    }, {})

    return resp.json()


def get_connection_by_id_v1(connection_id: str, credential: Optional[TokenCredential] = None) -> dict:
    """Get a connection from a workspace."""
    uri_match = re.match(r"/subscriptions/(.*)/resourceGroups/(.*)/" +
                         r"providers/Microsoft.MachineLearningServices/" +
                         r"workspaces/(.*)/connections/(.*)", connection_id)

    if uri_match is None:
        logger.warning(f"Invalid connection_id {connection_id}, expecting Azure Machine Learning resource ID")
        logger.warning("Trying with current Workspace credentials.")
        from run_utils import TestRun
        run = TestRun()
        subscriptio_id = run.subscription
        ws_name = run.workspace.name
        resource_group = run.workspace.resource_group
        connection_string = connection_id

    else:
        subscriptio_id = uri_match.group(1)
        resource_group = uri_match.group(2)
        ws_name = uri_match.group(3)
        connection_string = uri_match.group(4)

    from azureml.core import Workspace
    try:
        ws = Workspace(
            subscription_id=subscriptio_id,
            resource_group=resource_group,
            workspace_name=ws_name
        )
    except Exception as e:
        logger.warning(f"Could not get workspace '{ws_name}': {e}")
        raise ValueError(f"Could not get workspace '{ws_name}'") from e

    return get_connection_by_name_v2(ws, connection_string)


def send_put_request(url, headers, payload):
    """Send a PUT request."""
    with create_session_with_retry() as session:
        response = session.put(url, data=json.dumps(payload), headers=headers)
        # Raise an exception if the response contains an HTTP error status code
        response.raise_for_status()

    return response.json()


def create_connection_v2(
        workspace,
        name, category: str,
        target: str,
        auth_type: str,
        credentials: dict,
        metadata: str):
    """Create a connection in a workspace."""
    url = f"https://management.azure.com/subscriptions/{workspace.subscription_id}\
            /resourcegroups/{workspace.resource_group}/providers/Microsoft.MachineLearningServices/workspaces/\
            {workspace.name}/connections/{name}?api-version=2023-04-01-preview"

    resp = send_put_request(url, {
        "Authorization": f"Bearer {workspace._auth.get_token('https://management.azure.com/.default').token}",
        "content-type": "application/json"
    }, {
        "properties": {
            "category": category,
            "target": target,
            "authType": auth_type,
            "credentials": credentials,
            "metadata": metadata
        }
    })

    return resp
