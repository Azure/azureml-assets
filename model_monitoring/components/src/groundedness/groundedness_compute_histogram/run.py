# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Groundedness Metrics Component.

This file currently contains all the logic for the Groundedness Metrics
Component.
The reason for putting everything into a single file is that the executors
don't have access to other files by default.
The functionality to simplify this code injection is still being developed.
"""

import argparse
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Dict, Generator, List, Optional, Tuple, Union
from urllib.request import Request, urlopen

import jinja2
import json5
import pandas as pd
import requests
import tiktoken
from azure.ai.ml.identity import CredentialUnavailableError
from azure.ai.ml.identity._internal import _scopes_to_resource
from azure.core.credentials import AccessToken, TokenCredential
from azure.keyvault.secrets import SecretClient
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, udf
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from shared_utilities import io_utils

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

DEFAULT_INDENT = 2

RATING = "rating"
LABEL_KEYS = [RATING]
QUESTION = "question"
ANSWER = "answer"
CONTEXT = "context"


# ==================  HTTP Constants ==================
# Timeout per each request: 5min
HTTP_REQUEST_TIMEOUT = 300

# "patience", i.e. how many requests can fail before we alert
IGNORE_FAILED_REQUESTS_COUNT = 10

# Minimum number of requests before we use error_rate_threshold
MIN_REQUEST_COUNT = 3

# ================= Endpoint Constants =================
AZURE_TOKEN_REFRESH_INTERVAL = 600  # seconds
AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE = r"^(?=.{1,255}$)(?!-)[a-zA-Z0-9-]{1,63}(?<!-)(\.(?!-)[a-zA-Z0-9-]{1,63}(?<!-))*\.(inference\.ml|openai)\.azure\.com(/openai)?$"  # noqa: E501
AZURE_ENDPOINT_URL_PATTERN = "https://{}/v1/engines/davinci/chat/completions"
AZURE_OPENAI_API_URL_PATTERN = "https://{}/openai/deployments/{}/chat/completions"

# OpenAI API
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

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

ENDPOINT_PARAMS = [
    "authorization_header",
    "azure_endpoint_domain_name",
    "azure_openai_api_version",
    "api_call_delay_sec",
    "endpoint_type",
    "request_error_rate_threshold",
    "api_call_retry_backoff_factor",
    "api_call_retry_max_count",
    "api_call_max_parallel_count",
    "model",
    "authorization_vault_url",
    "authorization_secret_name",
    "authorization_type",
]

# ---

MIN_RATING = 1
MAX_RATING = 5

TASK_INJECTION_JINJA_TEMPLATE = """
{% for sample in input_samples %}
## Task #{{ sample.index }}:
{{ sample.encoded_json }}

{% endfor %}
"""

EXAMPLE_TASK_INPUT = {
    CONTEXT: "The Oscars are a popular award ceremony.",
    QUESTION: "Did Lord of the Rings win any Oscars?",
    ANSWER: "Lord of the Rings won 10 Oscars.",
}

EXAMPLE_TASK_OUTPUT = {
    RATING: MIN_RATING,
}

INITIAL_ANNOTATION_INSTRUCTIONS = f'Given the several tasks with {CONTEXT} and {QUESTION}, score the {ANSWER} between {MIN_RATING} to {MAX_RATING} (integers only), where {MIN_RATING} means "inconsistency" and {MAX_RATING} means "perfect consistency". Note that consistency measures whether the facts in the {ANSWER} are consistent with the facts in the {CONTEXT}. Consider whether the {ANSWER} does reproduce facts accurately and does not make up untrue information.'  # noqa: E501
FINAL_ANNOTATION_REMINDER = f'Reminder: The return values for each task should be dictionaries with the key "{RATING}". The value of "{RATING}" should be an integer between {MIN_RATING} and {MAX_RATING}.'  # noqa: E501


GROUNDING_ANNOTATION_GUIDELINES = "\n\n".join(
    [
        INITIAL_ANNOTATION_INSTRUCTIONS,
        "## Example Task #0",
        json.dumps(EXAMPLE_TASK_INPUT),
        "A good example response would be:",
        "## Example Task #0",
        json.dumps(EXAMPLE_TASK_OUTPUT),
        "# Tasks",
        TASK_INJECTION_JINJA_TEMPLATE,
        FINAL_ANNOTATION_REMINDER,
    ]
)

OUTPUT_SPLITTING_REGEX = r"[# ]*Task #*\d+:"

AUTHORIZATION = "Authorization"
BEARER = "Bearer"
API_KEY = "api-key"
AUTH_HEADERS = [AUTHORIZATION, API_KEY]


def _str2bool(val) -> bool:
    """
    Resolve boolean arguments if they are not given in the standard format.

    Arguments:
        val (bool or string): boolean argument type
    Output:
        bool: the desired value {True, False}

    """
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        if val.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif val.lower() in ("no", "false", "f", "n", "0"):
            return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _check_and_format_azure_endpoint_url(
    url_pattern, domain_pattern_re, domain, api_version, model
):
    domain = domain.strip()
    if not re.match(domain_pattern_re, domain):
        raise RuntimeError(f"Invalid Azure endpoint domain URL: {domain}.")

    url = url_pattern.format(domain, model)

    if api_version:
        url += f"?api-version={api_version}"

    return url


class _TokenScope(Enum):
    AZURE_ENDPOINT = "https://ml.azure.com"
    AZURE_OPENAI_API = "https://cognitiveservices.azure.com"


# --- The following is copied from the yet to be released azureml-featurestore.
# TODO: replace with import once it's released.
class AzureMLHoboSparkOnBehalfOfCredential(TokenCredential):
    """Authenticates a user via the on-behalf-of flow on Hobo Spark compute.

    This credential can only be used on
    `Azure Machine Learning Hobo Spark Compute.`
    during job execution when user request to run job during its identity.
    """

    def __init__(self, **kwargs):  # noqa: D107
        provider_type = os.environ.get("AZUREML_DATAPREP_TOKEN_PROVIDER")
        if provider_type != "sparkobo":
            # OBO identity isn't available in this environment
            self._credential = None
        self._credential = _AzureMLHoboSparkOnBehalfOfCredential(**kwargs)

    def get_token(self, *scopes, **kwargs):
        """Request an access token for `scopes`.

        This method is called automatically by Azure SDK clients.

        :param str scopes: desired scope for the access token.
            This credential allows only one scope per request.
        :rtype: azure.core.credentials.AccessToken
        :return: AzureML On behalf of credentials isn't available in the
            hosting environment
        :raises: ~azure.ai.ml.identity.CredentialUnavailableError
        """
        if not self._credential:
            raise CredentialUnavailableError(message=self.get_unavailable_message())

        return self._credential.get_token(*scopes, **kwargs)

    def get_unavailable_message(self) -> str:  # noqa: D102
        return "AzureML On Behalf of credentials not available in this " "environment."


class _AzureMLHoboSparkOnBehalfOfCredential(object):
    def __init__(self, **kwargs):
        if len(kwargs) > 0:
            env_key_from_kwargs = [
                "AZUREML_SYNAPSE_CLUSTER_IDENTIFIER",
                "AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT",
                "AZUREML_RUN_ID",
                "AZUREML_RUN_TOKEN_EXPIRY",
            ]
            for env_key in env_key_from_kwargs:
                if env_key in kwargs.keys():
                    os.environ[env_key] = kwargs[env_key]
                else:
                    raise Exception(
                        "Unable to initialize AzureMLHoboSparkOBOCredential "
                        "due to invalid arguments"
                    )
        else:
            from pyspark.sql import SparkSession

            try:
                spark = SparkSession.builder.getOrCreate()
            except Exception:  # noqa: B902
                raise Exception(
                    "Fail to get spark session, please check if spark "
                    "environment is set up."
                )

            spark_conf = spark.sparkContext.getConf()
            spark_conf_vars = {
                "AZUREML_SYNAPSE_CLUSTER_IDENTIFIER": "spark.synapse.clusteridentifier",  # noqa: E501
                "AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT": "spark.tokenServiceEndpoint",  # noqa: E501
            }
            for env_key, conf_key in spark_conf_vars.items():
                value = spark_conf.get(conf_key)
                if value:
                    os.environ[env_key] = value

        self.obo_service_endpoint = os.environ.get("AZUREML_OBO_SERVICE_ENDPOINT")
        self.token_service_endpoint = os.environ.get(
            "AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT"
        )
        self.obo_access_token = os.environ.get("AZUREML_OBO_CANARY_TOKEN")
        self.cluster_identifier = os.environ.get("AZUREML_SYNAPSE_CLUSTER_IDENTIFIER")
        self.subscription_id = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
        self.resource_group = os.environ.get("AZUREML_ARM_RESOURCEGROUP")
        self.workspace_name = os.environ.get("AZUREML_ARM_WORKSPACE_NAME")
        self.experiment_name = os.environ.get("AZUREML_ARM_PROJECT_NAME")
        self.run_id = os.environ.get("AZUREML_RUN_ID")
        self.oid = os.environ.get("OID")
        self.tid = os.environ.get("TID")

        if not self.obo_access_token:
            return None

    def get_token(self, *scopes, **kwargs) -> AccessToken:
        resource = _scopes_to_resource(*scopes)
        request_url = (
            f"https://{self.token_service_endpoint}/api/v1/proxy/obotoken"
            f"/v1.0/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{self.resource_group}"
            "/providers/Microsoft.MachineLearningServices/"
            f"workspaces/{self.workspace_name}/getuseraccesstokenforrun"
        )

        request_body = {
            "oboToken": self.obo_access_token,
            "oid": self.oid,
            "tid": self.tid,
            "resource": resource,
            "experimentName": self.experiment_name,
            "runId": self.run_id,
        }

        headers = {
            "Content-Type": "application/json;charset=utf-8",
            "x-ms-proxy-host": self.obo_service_endpoint,
            "obo-access-token": self.obo_access_token,
            "x-ms-cluster-identifier": self.cluster_identifier,
        }

        print("Attempting to get token from AzureML OBO service.")
        try:
            response = _send_request(request_url, request_body, headers)
            if response:
                response_dict = json.loads(response.read().decode("utf-8"))
                access_token = AccessToken(
                    response_dict["token"], int(time.time()) + 3600
                )
                print("Finished getting token from AzureML OBO service.")
                return access_token
            else:
                print(
                    "Failed to get token from AzureML OBO service. "
                    f"Invalid response: {response.__dict__}"
                )
                return None

        except Exception as e:  # noqa: B902
            print("Failing in auth while sending request: " f"{response.__dict__}")
            raise e


def _send_request(url, data=None, headers=None, method=None):
    args = {"url": url}
    if data:
        data = json.dumps(data)
        args["data"] = data.encode("utf8")
    if headers:
        args["headers"] = headers
    if method:
        # the default is GET if data is None, POST otherwise
        args["method"] = method

    try:
        return urlopen(Request(**args), timeout=5)
    except:  # noqa: E722
        raise Exception(f"failed while sending a request to {url} with data {data}.")


# END of copied code from azureml-featurestore


class _APITokenManager(ABC):
    def __init__(
        self,
        *,
        endpoint_type,
        auth_header,
        **kwargs,
    ):
        self.endpoint_type = endpoint_type
        self.credential = self.get_aad_credential()
        self.token = None
        self.auth_header = auth_header
        self.last_refresh_time = None

    def get_aad_credential(self):
        return AzureMLHoboSparkOnBehalfOfCredential(
            AZUREML_SYNAPSE_CLUSTER_IDENTIFIER=os.environ[
                "AZUREML_SYNAPSE_CLUSTER_IDENTIFIER"
            ],
            AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT=os.environ[
                "AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT"
            ],
            AZUREML_RUN_ID=os.environ["AZUREML_RUN_ID"],
            AZUREML_RUN_TOKEN_EXPIRY=os.environ["AZUREML_RUN_TOKEN_EXPIRY"],
        )

    @abstractmethod
    def get_token(self):
        pass


class _ManagedIdentityAPITokenManager(_APITokenManager):
    def __init__(
        self,
        *,
        endpoint_type,
        auth_header,
        token_scope,
        **kwargs,
    ):
        super().__init__(endpoint_type=endpoint_type, auth_header=auth_header)
        self.token_scope = token_scope

    def get_token(self):
        if (
            self.token is None
            or self.last_refresh_time is None
            or time.time() - self.last_refresh_time > AZURE_TOKEN_REFRESH_INTERVAL
        ):
            self.last_refresh_time = time.time()
            self.token = self.credential.get_token(self.token_scope.value).token

        return self.token


class _KeyVaultAPITokenManager(_APITokenManager):
    def __init__(
        self,
        *,
        endpoint_type,
        auth_header,
        vault_url,
        secret_name,
        **kwargs,
    ):
        super().__init__(endpoint_type=endpoint_type, auth_header=auth_header)

        # Get Open AI API key from Key Vault and set it
        secret_client = SecretClient(vault_url=vault_url, credential=self.credential)
        openai_api_secret = secret_client.get_secret(secret_name)
        self.token = openai_api_secret.value

    def get_token(self):
        return self.token


class _PromptData:
    """Class for storing prompt information."""

    def __init__(
        self,
        input_idx: List[int],
        input_examples: List[str],
        prompt: str,
        n_tokens_estimate: int,
    ):
        self.input_idx = input_idx
        self.input_examples = input_examples
        self.prompt = prompt
        self.n_tokens_estimate = n_tokens_estimate


class _Tokenizer:
    """Handle LLM tokenizing using the tiktoken library."""

    def __init__(self, model_name: str):
        self.model_name = model_name

        # Get fast tokenizer for model_name
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, input_str: str) -> int:
        # Count tokens, including special tokens like <|endofprompt|>
        return len(self.encoding.encode(input_str, allowed_special="all"))


class _InputSample:
    def __init__(self, index, encoded_json):
        self.index = index
        self.encoded_json = encoded_json


class _PromptBuilder:
    def __init__(
        self,
        labeling_guidelines: str,
        model_name: str,
        max_shots: int,
        max_tokens: int,
        min_input_examples: int = 1,
    ):
        """Batches prompts according to limits."""
        self.labeling_guidelines = jinja2.Template(
            labeling_guidelines, undefined=jinja2.StrictUndefined
        )
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_shots = max_shots
        self.min_input_examples = min_input_examples
        self.tokenizer = _Tokenizer(
            model_name=model_name,
        )

    def generate_prompts(
        self,
        input_data_df: DataFrame,
        max_inputs: int,
    ) -> Generator[_PromptData, None, None]:
        """Generate prompts from input_pool.

        Prioritizes fitting as many examples as possible into the prompt
        while staying below token limits.
        If an example can't fit into the prompt, then it is run on its own.

        Args:
            input_data_df (DataFrame): DataFrame of input examples
            max_inputs (int): maximum number of input examples to use

        Returns:
            prompt_data: list of PromptDatas containing the prompt,
                the index of input examples, and the number of tokens.
        """
        input_data = input_data_df.to_dict(orient="records")
        input_data_length = len(input_data)
        next_index = 0
        stop_index = min(max_inputs, input_data_length)

        while next_index < input_data_length:
            input_idx = list(range(next_index, stop_index))
            input_examples = input_data[next_index:stop_index]

            # Build prompt given input, shot, and token limits
            prompt, n_tokens, n_inputs = self.build_prompt_with_limits(
                input_examples,
            )

            # send prompt
            next_index += n_inputs
            stop_index = min(next_index + max_inputs, input_data_length)
            input_idx = input_idx[:n_inputs]
            input_examples = input_examples[:n_inputs]
            yield _PromptData(
                input_idx=input_idx,
                input_examples=input_examples,
                prompt=prompt,
                n_tokens_estimate=n_tokens,
            )

    def build_prompt_with_limits(
        self,
        input_data: List[Dict[str, str]],
    ) -> Tuple[str, int, int]:
        """Reduces batch examples until the prompt is within token limits.

        Reduces input_examples if maximum tokens are reached.

        Args:
            input_examples: list of input examples

        Returns:
            prompt: constructed prompt
            n_tokens: number of tokens within the prompt
            input_examples: number of input examples that can be fit
        """
        # build prompt with all examples
        prompt = self.build_prompt(input_data)
        n_tokens = self.tokenizer.count_tokens(prompt)

        # reduce input examples iteratively until minimum is hit
        while n_tokens > self.max_tokens and len(input_data) > self.min_input_examples:
            input_data = input_data[:-1]
            prompt = self.build_prompt(input_data)
            n_tokens = self.tokenizer.count_tokens(prompt)

        return prompt, n_tokens, len(input_data)

    @staticmethod
    def encode_example(
        example: Dict[str, Union[int, bool, str]],
        key_order: Optional[List[str]] = None,
        indent: Optional[int] = DEFAULT_INDENT,
    ) -> str:
        """
        Encode examples into JSON.

        :param example: example to encode
        :param key_order: ordering of keys printed to string
        :param indent: number of spaces indented at each level
        :return: encoded example
        """
        if key_order:
            example = OrderedDict([(key, example[key]) for key in key_order])

        # Dump JSON with keys double-quoted and final comma removed
        return json5.dumps(
            example, indent=indent, quote_keys=True, trailing_commas=False
        )

    def build_prompt(
        self,
        input_examples: List[Dict[str, str]],
    ) -> str:
        """Build prompt from input_examples.

        Encode examples into JSON format.

        Args:
            input_examples: list of input examples

        Returns:
            prompt: constructed prompt
        """
        # create input prompt using pattern

        input_samples = [
            _InputSample(
                index=i,
                encoded_json=self.encode_example(
                    input_example, [QUESTION, CONTEXT, ANSWER]
                ),
            )
            for (i, input_example) in enumerate(input_examples, start=1)
        ]

        return self.labeling_guidelines.render(
            input_samples=input_samples,
        )

    def split_output_examples(self, output_str: str) -> List[str]:
        """Attempt to split the output into a list of examples.

        Args:
            output_str: output examples

        Returns:
            output_examples: list of output examples
        """
        output_str = output_str.strip()
        output_examples = [
            ex.strip()
            for ex in re.split(OUTPUT_SPLITTING_REGEX, output_str)
            if ex.strip()
        ]
        return output_examples


class _HTTPClientWithRetry:
    def __init__(self, n_retry, backoff_factor):
        self.attempts = n_retry

        retry_strategy = Retry(
            total=n_retry,
            status_forcelist=[104, 408, 409, 424, 429, 500, 502, 503, 504],
            backoff_factor=backoff_factor,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.client = requests.Session()
        self.client.mount("https://", adapter)


def _request_api(
    endpoint_type,
    session,
    endpoint_url: str,
    token_manager: _APITokenManager,
    **request_params,
):
    """Make REST call to API and return parsed result."""
    token = token_manager.get_token()

    headers = {
        "Content-Type": "application/json",
    }

    print(f"Using {token_manager.auth_header} authentication")
    if token_manager.auth_header == BEARER:
        headers[AUTHORIZATION] = f"{BEARER} {token}"
    elif token_manager.auth_header == API_KEY:
        headers[API_KEY] = token

    time_start = time.time()

    # Update timeout for proxy endpoint
    if endpoint_type == "azure_endpoint":
        headers["timeout_ms"] = "90000"

    # print headers without disclosing token
    headers_output = {
        h: (headers[h] if h not in AUTH_HEADERS else "*" * len(headers[h]))
        for h in headers
    }
    print(
        f"Sending request \n    to endpoint: {endpoint_url}"
        f"\n    with headers: {headers_output}"
        f"\n    and params: {request_params}"
    )
    with session.post(
        endpoint_url,
        headers=headers,
        json=request_params,
        timeout=HTTP_REQUEST_TIMEOUT,
    ) as response:
        time_taken = str(time.time() - time_start)
        print(f"Received response from endpoint: {response.__dict__}")
        print(f"Time taken to receive response: {time_taken}")
        if response.status_code == 200:
            response_data = response.json()

            time_taken = time.time() - time_start
            parsed_response = {
                "samples": [r["message"]["content"] for r in response_data["choices"]],
                "finish_reason": [r["finish_reason"] for r in response_data["choices"]],
            }

            return parsed_response, time_taken

        else:
            raise Exception(
                "Received unexpected HTTP status: "
                f"{response.status_code} {response.text}"
            )


class _Job:
    """Job state for prompts submitted to model engines."""

    def __init__(
        self,
        job_idx: int,
        prompt_data: _PromptData,
        request_params: dict,
        retries_attempted: int = 0,
        response_data: dict = None,
        status: str = None,
    ):
        self.job_idx = job_idx
        self.prompt_data = prompt_data
        self.request_params = request_params
        self.retries_attempted = retries_attempted
        self.response_data = response_data or {}
        self.status = status

    def get_output_data(self) -> dict:
        """Return the data which gets written to output."""
        return {
            **(self.prompt_data.__dict__),
            **self.response_data,
            "retries": self.retries_attempted,
            "status": self.status,
        }


def _query_inference_for_line(
    jobs: List[_Job],
    session: requests.Session,
    endpoint_url: str,
    token_manager: _APITokenManager,
    request_error_rate_threshold: float,
    api_call_delay_sec: float,
    endpoint_type: str,
):
    # if we count too many errors, we stop and raise an exception
    error_count = 0
    request_count = 0
    outputs = []
    for job in jobs:
        request_count += 1

        # Copy request_data to avoid modifying the original dict.
        request_data = {
            "model": job.request_params["model"],
            "temperature": job.request_params["temperature"],
            "top_p": job.request_params["top_p"],
            "n": job.request_params["num_samples"],
            "max_tokens": job.request_params["max_tokens"],
            "frequency_penalty": job.request_params["frequency_penalty"],
            "presence_penalty": job.request_params["presence_penalty"],
            "messages": [{"role": "user", "content": job.prompt_data.prompt}],
        }

        if job.request_params.get("stop", None):
            request_data["stop"] = job.request_params["stop"]

        response = {}
        try:
            response, time_taken = _request_api(
                endpoint_type=endpoint_type,
                session=session,
                endpoint_url=endpoint_url,
                token_manager=token_manager,
                **request_data,
            )

            # Append time taken to the line
            response["response_time_sec"] = time_taken

        except Exception as e:  # noqa: B902
            response["finish_reason"] = ["error"]
            response["error"] = [str(e)]

            error_count += 1
            error_rate = error_count / request_count

            # if we count too many errors, we stop and raise an exception
            if error_count > IGNORE_FAILED_REQUESTS_COUNT or (
                request_count >= MIN_REQUEST_COUNT
                and error_rate >= request_error_rate_threshold
            ):
                raise e

        job.response_data = response
        outputs.append(job)

        # Sleep between consecutive requests to avoid rate limit
        time.sleep(api_call_delay_sec)
    return outputs


def _request_prompt_batch(
    jobs: List[_Job],
    token_manager: _APITokenManager,
    model: str,
    azure_endpoint_domain_name: str,
    endpoint_type: str,
    azure_openai_api_version: str,
    request_error_rate_threshold: float = 0.5,
    api_call_delay_sec: float = 0.5,
    api_call_retry_backoff_factor: int = 3,
    api_call_retry_max_count: int = 3,
    api_call_max_parallel_count: int = 1,
    **kwargs,
) -> List[_Job]:
    # Check arguments
    if endpoint_type == "openai_api":  # Prevent exceeding API limits
        if api_call_max_parallel_count > 1:
            raise RuntimeError(
                "OpenAI API inference supports only 1 parallel request to "
                "avoid rate limit. Received "
                f"api_call_max_parallel_count={api_call_max_parallel_count}"
            )
        if api_call_delay_sec < 3 or api_call_retry_backoff_factor < 3:
            raise RuntimeError(
                "OpenAI API inference must use at least 3-second delay "
                "between requests to avoid rate limit. "
                f"Received api_call_delay_sec={api_call_delay_sec} and "
                f"api_call_retry_backoff_factor={api_call_retry_backoff_factor}"  # noqa: E501
            )

    # Set endpoint URL
    if endpoint_type == "openai_api":
        endpoint_url = OPENAI_API_URL
    else:
        if endpoint_type == "azure_endpoint":
            url_pattern = AZURE_ENDPOINT_URL_PATTERN
        elif endpoint_type == "azure_openai_api":
            url_pattern = AZURE_OPENAI_API_URL_PATTERN

        azure_endpoint_url = _check_and_format_azure_endpoint_url(
            url_pattern,
            AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE,
            azure_endpoint_domain_name,
            azure_openai_api_version,
            model,
        )
        endpoint_url = azure_endpoint_url

    print(f"Determined endpoint URL {endpoint_url}")

    httpClient = _HTTPClientWithRetry(
        n_retry=api_call_retry_max_count,
        backoff_factor=api_call_retry_backoff_factor,
    )

    with httpClient.client as session:
        if len(jobs) == 0:
            return []  # queue is empty

        return _query_inference_for_line(
            jobs,
            session,
            endpoint_url,
            token_manager,
            request_error_rate_threshold,
            api_call_delay_sec,
            endpoint_type,
        )


class _JobManager:
    """Handles batching and job state management."""

    def __init__(
        self,
        prompt_builder: _PromptBuilder,
    ):
        self.prompt_builder = prompt_builder
        self.job_idx = 0

    def submit_inputs(
        self,
        input_data_df: DataFrame,
        request_args: dict,
        max_inputs: int,
        num_samples: int,
        **endpoint_args,
    ) -> DataFrame:
        """Run inference over input dataset.

        Args:
            input_data_df: Spark DataFrame with input data.
            request_args (dict): Arguments to pass to API.
            max_inputs (int): Maximum number of inputs per prompt.
            num_samples (int): Number of samples to generate per prompt.
            ``**endpoint_args``: Arguments passed to _request_prompt_batch
        """
        print(f"starting submit_inputs with endpoint_args={endpoint_args}")
        # Define authorization token manager
        authorization_type = endpoint_args["authorization_type"]
        endpoint_type = endpoint_args["endpoint_type"]
        auth_header = endpoint_args["authorization_header"]
        if authorization_type == "key_vault_secret":
            token_manager_class = _KeyVaultAPITokenManager
            token_scope = None

        elif authorization_type == "managed_identity":
            token_manager_class = _ManagedIdentityAPITokenManager

            if endpoint_type == "azure_endpoint":
                token_scope = _TokenScope.AZURE_ENDPOINT
            elif endpoint_type == "azure_openai_api":
                token_scope = _TokenScope.AZURE_OPENAI_API

        token_manager = token_manager_class(
            endpoint_type=endpoint_type,
            vault_url=endpoint_args.get("authorization_vault_url"),
            secret_name=endpoint_args.get("authorization_secret_name"),
            token_scope=token_scope,
            auth_header=auth_header,
        )

        print(
            "Created token manager for auth type "
            f"{authorization_type} using auth header {auth_header}."
        )

        # Build batched prompts using inputs
        prompts = self.prompt_builder.generate_prompts(
            input_data_df,
            max_inputs=max_inputs,
        )
        jobs = [
            self.make_job(prompt_data=prompt_data, request_params=request_args)
            for prompt_data in prompts
        ]

        prompts_to_print = "\n".join(
            [f"    Prompt: {job.prompt_data.prompt}" for job in jobs]
        )
        print(f"Generated prompts: \n{prompts_to_print}\n")

        responses = _request_prompt_batch(
            jobs=jobs, token_manager=token_manager, **endpoint_args
        )

        responses_to_print = "\n".join(
            [f"    Response: {job.response_data}" for job in responses]
        )
        print(f"Received responses from annotation: \n{responses_to_print}\n")

        # Parse responses for this batch of jobs
        parsed_jobs = self.parse_responses(responses, num_samples)

        # TODO handle errors and attempt retries
        job_results = [
            result
            for job in parsed_jobs
            if job.status == "success"
            for result in job.response_data["output_examples"][0]
        ]
        return pd.DataFrame(
            {key: [res[key] for res in job_results] for key in LABEL_KEYS}
        )

    @staticmethod
    def decode_example(
        example: str,
        label_keys: List[str],
    ) -> Dict[str, Union[int, bool, str]]:
        """Decode example from an encoding format.

        :param example: example to decode
        :param key_order: ordering of keys printed to string
        :return: decoded example
        """
        example = example.strip()
        start = example.find("{")
        end = example.rfind("}")
        if start == -1:
            raise ValueError("Could not find starting curly brace.")
        if end == -1:
            raise ValueError("Could not find ending curly brace.")

        example = json5.loads(example[start: end + 1])

        # check if label keys are in example
        for label_key in label_keys:
            if label_key not in example:
                raise ValueError(f"Label key {label_key} not in output example")

        return example

    def parse_responses(
        self,
        responses: List[_Job],
        num_samples: int,
    ) -> Generator[_Job, None, None]:
        """Given a stream of job responses, parse them."""
        for job in responses:
            label_keys = LABEL_KEYS
            job.response_data["label_keys"] = label_keys

            n_inputs = len(job.prompt_data.input_idx)

            if job.response_data["finish_reason"] == ["error"]:
                job.status = "endpoint error"
                print(
                    "Cannot parse job due to finish_reason=error: "
                    f"{job.response_data}"
                )
                continue

            assert len(job.response_data["samples"]) == num_samples, (
                f"Expected {num_samples} samples, "
                f"got {len(job.response_data['samples'])}"
            )

            output_examples = []
            num_failed = 0
            for sample in job.response_data["samples"]:
                # try to split the output into {num_samples} examples
                try:
                    sample_examples = self.prompt_builder.split_output_examples(sample)

                    if len(sample_examples) < n_inputs:
                        raise ValueError(
                            f"Expected at least {n_inputs} examples, "
                            f"but got {len(sample_examples)}"
                        )

                    sample_examples = sample_examples[:n_inputs]
                except Exception:  # noqa: B902, F841
                    output_examples.append(None)
                    print(f"Failed splitting output into examples: {sample}")
                    num_failed += 1
                    continue

                # try to decode each example and check for the label keys
                try:
                    sample_examples_parsed = []
                    for example in sample_examples:
                        sample_examples_parsed.append(
                            self.decode_example(example, label_keys)
                        )
                    output_examples.append(sample_examples_parsed)
                except Exception as _:  # noqa: B902, F841
                    output_examples.append(None)
                    print(f"Failed decoding examples: {sample_examples}")
                    num_failed += 1

            if num_failed == num_samples:
                job.status = "parsing failed"
                yield job
            else:
                job.status = "success"
                job.response_data["output_examples"] = output_examples
                yield job

    def make_job(self, **job_params) -> _Job:
        """Make job, tracking job_idx internally."""
        new_job = _Job(job_idx=self.job_idx, **job_params)
        self.job_idx += 1
        return new_job


def run():
    """Compute metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dataset", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_deployment_name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--stop", type=str, default=None)
    parser.add_argument(
        "--endpoint_type",
        type=str,
        required=False,
        choices=["openai_api", "azure_endpoint", "azure_openai_api"],
        default="azure_openai_api",
    )
    parser.add_argument("--azure_endpoint_domain_name", type=str, required=True)
    parser.add_argument(
        "--authorization_type",
        type=str,
        required=True,
        choices=["managed_identity", "key_vault_secret"],
    )
    parser.add_argument("--authorization_vault_url", type=str, required=False)
    parser.add_argument("--authorization_secret_name", type=str, required=False)
    parser.add_argument("--azure_openai_api_version", type=str, default="")
    parser.add_argument("--sample_rate", type=float, required=False, default=1.0)
    parser.add_argument("--api_call_max_parallel_count", type=int, default=1)
    parser.add_argument(
        "--request_error_rate_threshold",
        type=float,
        default=0.5,
        help="Fail if the running error rate for the endpoint requests "
        "raises above this threshold.",
    )
    parser.add_argument("--api_call_delay_sec", type=float, default=0.5)
    parser.add_argument("--api_call_retry_backoff_factor", type=int, default=3)
    parser.add_argument("--api_call_retry_max_count", type=int, default=10)
    parser.add_argument("--histogram", type=str, required=True)
    args = parser.parse_args()

    request_args = {
        arg: getattr(args, arg) for arg in OPENAI_REQUEST_PARAMS if hasattr(args, arg)
    }
    endpoint_args = {
        arg: getattr(args, arg) for arg in ENDPOINT_PARAMS if hasattr(args, arg)
    }
    # add model to both request and endpoint args
    # The arg name is longer to be as explicit as possible.
    request_args["model"] = args.model_deployment_name
    endpoint_args["model"] = args.model_deployment_name

    # base definition: args.model_type == "gpt-35-turbo":
    base_max_tokens = 1000
    base_max_inputs = 10
    base_max_prompt_tokens = 3000
    model_type_factor = {"gpt-35-turbo": 1, "gpt-4": 2, "gpt-4-32k": 8}
    request_args["max_tokens"] = base_max_tokens * model_type_factor[args.model_type]
    max_inputs = base_max_inputs * model_type_factor[args.model_type]
    max_prompt_tokens = base_max_prompt_tokens * model_type_factor[args.model_type]

    print(f"Using max_inputs = {max_inputs}")
    print(f"Using max_prompt_tokens = {max_prompt_tokens}")

    if args.authorization_type == "managed_identity":
        endpoint_args["authorization_header"] = BEARER
    else:  # args.authorization_type == "key_vault_secret"
        endpoint_args["authorization_header"] = API_KEY

    # Validate inputs
    if args.temperature < 0.0 or args.temperature > 2.0:
        raise ValueError(f"temperature must be in [0.0, 2.0), got {args.temperature}.")
    if args.top_p < 0.0 or args.top_p > 1.0:
        raise ValueError(f"top_p must be in [0.0, 1.0], got {args.top_p}.")
    if args.num_samples <= 0:
        # TODO support multiple returned annotations
        raise ValueError(f"num_samples must be 1, got {args.num_samples}.")
    if args.frequency_penalty < -2.0 or args.frequency_penalty > 2.0:
        raise ValueError(
            "frequency_penalty must be in (-2.0, 2.0), "
            f"got {args.frequency_penalty}."
        )
    if args.presence_penalty < -2.0 or args.presence_penalty > 2.0:
        raise ValueError(
            "presence_penalty must be in (-2.0, 2.0), " f"got {args.presence_penalty}."
        )
    if args.endpoint_type != "azure_openai_api":
        # TODO: support other endpoint types
        raise ValueError("endpoint_type must be azure_openai_api.")
    if args.authorization_type == "key_vault_secret" and (
        args.authorization_vault_url is None or args.authorization_secret_name is None
    ):
        raise ValueError(
            "authorization_vault_url and authorization_secret_name must be "
            "provided for key_vault_secret authorization_type."
        )
    elif args.authorization_type == "managed_identity":
        # TODO support managed_identity authorization_type
        raise ValueError("managed_identity authorization_type is not supported yet.")
    if args.sample_rate <= 0.0 or args.sample_rate > 1.0:
        raise ValueError(f"sample_rate must be in (0.0, 1.0], got {args.sample_rate}.")

    print(f"Running with args: {args}")

    assess_groundedness(
        target_dataset=args.target_dataset,
        histogram=args.histogram,
        max_inputs=max_inputs,
        max_prompt_tokens=max_prompt_tokens,
        model=args.model_deployment_name,
        num_samples=args.num_samples,
        sample_rate=args.sample_rate,
        request_args=request_args,
        endpoint_args=endpoint_args,
    )


def assess_groundedness(
    *,
    target_dataset,
    histogram,
    max_inputs,
    max_prompt_tokens,
    model,
    num_samples,
    sample_rate,
    request_args,
    endpoint_args,
):
    """Assess groundedness."""
    target_df = io_utils.read_mltable_in_spark(target_dataset)
    for col_name in [QUESTION, ANSWER, CONTEXT]:
        if col_name not in target_df.columns:
            raise ValueError(f"target_dataset must have column: {col_name}")

    # Sampling
    target_df = target_df.sample(withReplacement=False, fraction=sample_rate)

    spark = io_utils.init_spark()
    spark_conf = spark.sparkContext.getConf()
    spark_conf_vars = {
        "AZUREML_SYNAPSE_CLUSTER_IDENTIFIER": "spark.synapse.clusteridentifier",  # noqa: E501
        "AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT": "spark.tokenServiceEndpoint",
    }
    for env_key, conf_key in spark_conf_vars.items():
        value = spark_conf.get(conf_key)
        if value:
            os.environ[env_key] = value

    driver_env_vars = {
        k: v
        for k, v in os.environ.items()
        if k
        in [
            "AZUREML_SYNAPSE_CLUSTER_IDENTIFIER",
            "AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT",
            "AZUREML_RUN_ID",
            "AZUREML_RUN_TOKEN_EXPIRY",
            "AZUREML_OBO_SERVICE_ENDPOINT",
            "AZUREML_OBO_CANARY_TOKEN",
            "AZUREML_ARM_SUBSCRIPTION",
            "AZUREML_ARM_RESOURCEGROUP",
            "AZUREML_ARM_WORKSPACE_NAME",
            "AZUREML_ARM_PROJECT_NAME",
            "OID",
            "TID",
        ]
    }

    # Run inference over input dataset
    def annotate_batch(iterator):
        # _PromptBuilder can iteratively batch unlabeled examples
        prompt_builder = _PromptBuilder(
            labeling_guidelines=GROUNDING_ANNOTATION_GUIDELINES,
            model_name=model,
            max_tokens=max_prompt_tokens,
            max_shots=0,
            min_input_examples=1,
        )

        # _JobManager handles batching and retry logic
        job_manager = _JobManager(
            prompt_builder=prompt_builder,
        )
        for batch in iterator:
            # add environment variables on executors
            for env_var_key, env_var_value in driver_env_vars.items():
                os.environ[env_var_key] = env_var_value

            print("Copied environment variables from driver to executor.")

            yield job_manager.submit_inputs(
                input_data_df=batch,
                request_args=request_args,
                max_inputs=max_inputs,
                num_samples=num_samples,
                **endpoint_args,
            )

    annotations_df = target_df.mapInPandas(
        annotate_batch,
        schema=StructType(
            [
                StructField(name=RATING, dataType=IntegerType(), nullable=True),
            ]
        ),
    )

    # Get rating counts
    metrics_df = annotations_df.select(RATING).groupBy(RATING).count()
    metrics_df.show()
    print("Finished annotating answers.")

    # if any buckets were empty, add them with count of 0
    for rating in range(MIN_RATING, MAX_RATING + 1):
        if metrics_df.filter(col(RATING) == rating).count() == 0:
            metrics_df = metrics_df.union(
                spark.createDataFrame([(rating, 0)], metrics_df.schema)
            )

    def _map_to_groundedness_bucket_name(rating):
        return f"GroundednessCount_{rating}"

    _map_to_groundedness_bucket_name_udf = udf(
        _map_to_groundedness_bucket_name, StringType()
    )

    # add metric_name, metric_value, data_type, and feature_name columns
    metrics_df = (
        metrics_df.withColumnRenamed("count", "metric_value")
        .withColumn("data_type", lit("numerical"))
        .withColumn("metric_name", _map_to_groundedness_bucket_name_udf(col(RATING)))
        .select("metric_name", "metric_value", "data_type")
        .withColumn("feature_name", lit(""))
    )
    # add total row count
    metrics_df = metrics_df.union(
        spark.createDataFrame(
            [("TargetRowCount", annotations_df.count(), "numerical", "")],
            metrics_df.schema,
        )
    )

    print("Finished calculating metrics based on annotations.")

    io_utils.save_spark_df_as_mltable(metrics_df, histogram)


if __name__ == "__main__":
    run()
