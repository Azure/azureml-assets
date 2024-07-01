# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LLM helper functions."""

import os
import re
import time
from abc import abstractmethod, ABC
import requests
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


# Parameters to OpenAI API requests
OPENAI_REQUEST_PARAMS = [
    "messages",  # prompt is only a param with the completions API
    "max_tokens",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "model",
    "num_samples",
]

AUTHORIZATION = "Authorization"
BEARER = "Bearer"
API_KEY = "api-key"
TEMPERATURE_VALUE = 0.0
TOP_P_VALUE = 1.0
NUM_SAMPLES_VALUE = 1
FREQUENCY_PENALTY_VALUE = 0.0
PRESENCE_PENALTY_VALUE = 0.0
STOP_VALUE = None

# Timeout per each request: 5min
HTTP_REQUEST_TIMEOUT = 300

# ================= Endpoint Constants =================
AZURE_TOKEN_REFRESH_INTERVAL = 600  # seconds
AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE = r"^(?=.{1,255}$)(?!-)[a-zA-Z0-9-]{1,63}(?<!-)(\.(?!-)[a-zA-Z0-9-]{1,63}(?<!-))*\.(inference\.ml|openai)\.azure\.com(/openai)?$"  # noqa: E501
AZURE_OPENAI_API_COMPLETION_URL_PATTERN = "https://{}/openai/deployments/{}/chat/completions"
LISTSECRETS_API_PATTERN = "https://management.azure.com{}/listsecrets?api-version=2023-06-01-preview"
AZURE_OPENAI_API_DEPLOYMENT_URL_PATTERN = "https://{}/openai/deployments/{}"


class _APITokenManager(ABC):
    def __init__(
        self,
        *,
        auth_header,
        **kwargs,
    ):
        self.credential = self.get_aad_credential()
        self.token = None
        self.auth_header = auth_header
        self.last_refresh_time = None

    def get_aad_credential(self):
        return AzureMLOnBehalfOfCredential()

    @abstractmethod
    def get_token(self):
        pass


class _WorkspaceConnectionTokenManager(_APITokenManager):
    def __init__(
        self,
        *,
        connection_name,
        auth_header,
        **kwargs,
    ):
        super().__init__(auth_header=auth_header)

        try:
            from azureml.dataprep.api._aml_auth._azureml_token_authentication import AzureMLTokenAuthentication
            from azure.ai.ml import MLClient
            from azure.ai.ml.entities import WorkspaceConnection

            credential = AzureMLTokenAuthentication._initialize_aml_token_auth()

            uri_match = re.match(r"/subscriptions/(.*)/resourceGroups/(.*)/providers/Microsoft.MachineLearningServices/workspaces/(.*)/connections/(.*)",  # noqa: E501
                                 connection_name, flags=re.IGNORECASE)

            ml_client = MLClient(
                credential=credential,
                subscription_id=uri_match.group(1),
                resource_group_name=uri_match.group(2),
                workspace_name=uri_match.group(3)
            )

            if os.environ.get("AZUREML_RUN_ID", None) is not None:
                # In AzureML Run context, we need to use workspaces internal endpoint that will accept
                # AzureMLToken auth.
                ml_client.connections._operation._client._base_url = f"{os.environ.get('AZUREML_SERVICE_ENDPOINT')}/rp/workspaces"  # noqa: E501
                print(f"Using ml_client base_url: {ml_client.connections._operation._client._base_url}")
                list_secrets_response = ml_client.connections._operation.list_secrets(
                    connection_name=uri_match.group(4),
                    resource_group_name=ml_client.resource_group_name,
                    workspace_name=ml_client.workspace_name,
                )
                connection = WorkspaceConnection._from_rest_object(list_secrets_response)
                print(f"Retrieved Workspace Connection: {connection.id}")

                if connection.type != "azure_open_ai":
                    raise Exception(f"Received unexpected endpoint type {connection.type}"
                                    "only Azure Open AI endpoints are supported at this time")
                api_version = "2023-07-01-preview"
                if hasattr(connection.metadata, "ApiVersion"):
                    api_version = connection.metadata["ApiVersion"]
                self.api_version = api_version
                self.domain_name = connection.target
                self.token = connection.credentials["key"]
            else:
                raise Exception("Unable to retrieve the token to establish a Workspace Connection")
        except Exception as e:
            raise Exception(f"Error encountered while attempting to authentication token: {e}")

    def get_api_version(self):
        return self.api_version

    def get_endpoint_domain(self):
        return self.domain_name

    def get_token(self):
        return self.token


class _HTTPClientWithRetry:
    def __init__(self, n_retry, backoff_factor):
        self.attempts = n_retry

        retry_strategy = Retry(
            total=n_retry,
            status=n_retry,
            status_forcelist=[104, 408, 409, 424, 429, 500, 502, 503, 504],
            backoff_factor=backoff_factor,
            respect_retry_after_header=True,
            allowed_methods=frozenset(['GET', 'POST', 'PUT']),
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.client = requests.Session()
        self.client.mount("https://", adapter)


def _request_api(session,
                 endpoint_url,
                 token_manager,
                 **request_params):
    """Make REST call to API and return parsed result."""
    token = token_manager.get_token()

    headers = {
        "Content-Type": "application/json",
    }

    # print(f"Using {token_manager.auth_header} authentication")
    if token_manager.auth_header == BEARER:
        headers[AUTHORIZATION] = f"{BEARER} {token}"
    elif token_manager.auth_header == API_KEY:
        headers[API_KEY] = token

    time_start = time.time()

    # print headers without disclosing token
    # headers_output = {
    #     h: (headers[h] if h not in [AUTHORIZATION] else "*" * len(headers[h]))
    #     for h in headers
    # }
    # print(
    #     f"Sending request \n    to endpoint: {endpoint_url}"
    #     f"\n    with headers: {headers_output}"
    #     f"\n    and params: {request_params}"
    # )
    response = session.post(
        endpoint_url,
        headers=headers,
        json=request_params,
        timeout=HTTP_REQUEST_TIMEOUT)
    time_taken = str(time.time() - time_start)
    # print(f"Received response from endpoint: {response.__dict__}")
    # print(f"Time taken to receive response: {time_taken}")
    if response.status_code == 200:
        response_data = response.json()

        time_taken = time.time() - time_start
        parsed_response = (
            {
                "samples": [
                    r["message"]["content"]
                    for r in response_data["choices"]
                ],
                "index": request_params,
                "finish_reason": [
                    r["finish_reason"] for r in response_data["choices"]
                ],
            }
        )
        return parsed_response, time_taken

    else:
        raise Exception(
            "Received unexpected HTTP status: "
            f"{response.status_code} {response.text}"
        )


def _check_and_format_azure_endpoint_url(url_pattern, domain_pattern_re, domain, api_version, model):
    domain = domain.strip()
    if domain.endswith('/'):
        domain = domain[:-1]

    if not re.match(domain_pattern_re, domain):
        raise RuntimeError(f"Invalid Azure endpoint domain URL: {domain}.")

    url = url_pattern.format(domain, model)

    if api_version:
        url += f"?api-version={api_version}"

    return url


def get_openai_request_args(args):
    """Get openai request parameters."""
    request_args = {
        arg: getattr(args, arg) for arg in OPENAI_REQUEST_PARAMS if hasattr(args, arg)
    }
    request_args["model"] = args.model_deployment_name
    return request_args


def get_llm_request_args(model_deployment_name):
    """Get openai request parameters."""
    request_args = {
        "temperature": TEMPERATURE_VALUE,
        "top_p": TOP_P_VALUE,
        "num_samples": NUM_SAMPLES_VALUE,
        "frequency_penalty": FREQUENCY_PENALTY_VALUE,
        "presence_penalty": PRESENCE_PENALTY_VALUE,
        "model": model_deployment_name
    }
    request_args["model"] = model_deployment_name
    return request_args
