import os
import argparse
import re
import json
import time
from abc import abstractmethod, ABC
from typing import List, Tuple
import requests
from azure.core.credentials import AccessToken, TokenCredential
from azure.ai.ml.identity import CredentialUnavailableError
from azure.ai.ml.identity._internal import _scopes_to_resource
from urllib.request import Request, urlopen
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

RATING = "rating"
INDEX = "index"
LABEL_KEYS = [RATING]
PROMPT = "prompt"
COMPLETION = "completion"
CONTEXT = "context"
GROUND_TRUTH = "ground_truth"

MIN_RATING = 1
MAX_RATING = 5

AUTHORIZATION = "Authorization"
BEARER = "Bearer"
API_KEY = "api-key"

# Timeout per each request: 5min
HTTP_REQUEST_TIMEOUT = 300

# ================= Endpoint Constants =================
AZURE_TOKEN_REFRESH_INTERVAL = 600  # seconds
AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE = r"^(?=.{1,255}$)(?!-)[a-zA-Z0-9-]{1,63}(?<!-)(\.(?!-)[a-zA-Z0-9-]{1,63}(?<!-))*\.(inference\.ml|openai)\.azure\.com(/openai)?$"  # noqa: E501
AZURE_OPENAI_API_COMPLETION_URL_PATTERN = "https://{}/openai/deployments/{}/chat/completions"
LISTSECRETS_API_PATTERN = "https://management.azure.com{}/listsecrets?api-version=2023-06-01-preview"
AZURE_OPENAI_API_DEPLOYMENT_URL_PATTERN = "https://{}/openai/deployments/{}"


RELEVANCE_ANNOTATION_TEMPLATE = "\n\n".join(
    [
        "System:",
        f"You are an AI assistant. You will be given the definition of an evaluation metric for assessing the \
            quality of an {COMPLETION} in a question-answering task. Your job is to compute an accurate evaluation \
                score using the provided evaluation metric.",
        f"Relevance measures how well the {COMPLETION} addresses the main aspects of the {PROMPT}. Consider whether \
            all and only the important aspects are contained in the {COMPLETION} when \
                evaluating relevance. Given the {PROMPT}, score the relevance of the {COMPLETION} \
                    between {MIN_RATING} to {MAX_RATING} using the following {RATING} scale:",
        f"{RATING} 1: the answer completely lacks relevance",
        f"{RATING} 2: the answer mostly lacks relevance",
        f"{RATING} 3: the answer is partially relevant",
        f"{RATING} 4: the answer is mostly relevant",
        f"{RATING} 5: the answer has perfect relevance",
        f"The score should be integer only, between {MIN_RATING} and {MAX_RATING}.",
        "## Example Task #0",
        json.dumps({
            PROMPT: "What field did Marie Curie excel in?",
            COMPLETION: "Marie Curie was a renowned painter who focused mainly on impressionist styles and \
                techniques.",
        }),
        "A good example response would be:",
        "## Example Task #0:",
        json.dumps({
            RATING: 1,
        }),
        "## Example Task #1",
        json.dumps({
            PROMPT: "Where were The Beatles formed?",
            COMPLETION: "The band The Beatles began their journey in London, England, and they changed the \
                history of music.",
        }),
        "A good example response would be:",
        "## Example Task #1:",
        json.dumps({
            RATING: 2,
        }),
        "## Example Task #2",
        json.dumps({
            PROMPT: "What are the main goals of Perseverance Mars rover mission?",
            COMPLETION: "The Perseverance Mars rover mission focuses on searching for signs of ancient life on Mars.",
        }),
        "A good example response would be:",
        "## Example Task #2",
        json.dumps({
            RATING: 3,
        }),
        "## Example Task #3",
        json.dumps({
            PROMPT: "What are the main components of the Mediterranean diet?",
            COMPLETION: "The Mediterranean diet primarily consists of fruits, vegetables, whole grains, and legumes.",
        }),
        "A good example response would be:",
        "## Example Task #3:",
        json.dumps({
            RATING: 4,
        }),
        "## Example Task #4",
        json.dumps({
            PROMPT: "What are the main attractions of the Queen's Royal Castle?",
            COMPLETION: "The main attractions of the Queen's Royal Castle are its expansive 500-acre grounds, \
                extensive gardens, parks, and the historical castle itself, which dates back to the 15th century \
                    and has housed generations of royalty.",
        }),
        "A good example response would be:",
        "## Example Task #4:",
        json.dumps({
            RATING: 5,
        }),
        "User:",
        "{input_samples}"
    ]
)


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
        return "AzureML On Behalf of credentials not available in this environment."


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
                "AZUREML_SYNAPSE_CLUSTER_IDENTIFIER": "spark.synapse.clusteridentifier",
                "AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT": "spark.tokenServiceEndpoint",
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
            f"workspaces/{self.workspace_name}/getuseraccesstokenforspark"
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
            print(f"Failing in auth while sending request: {response.__dict__}")
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
        raise Exception(f"Failed while sending a request to {url} with data {data}.")


# END of copied code from azureml-featurestore

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
        return AzureMLHoboSparkOnBehalfOfCredential(
            AZUREML_SYNAPSE_CLUSTER_IDENTIFIER=os.environ.get(
                "AZUREML_SYNAPSE_CLUSTER_IDENTIFIER", "spark.synapse.clusteridentifier"
            ),
            AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT=os.environ.get(
                "AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT", "spark.tokenServiceEndpoint"
            ),
            AZUREML_RUN_ID=os.environ["AZUREML_RUN_ID"],
            AZUREML_RUN_TOKEN_EXPIRY=os.environ["AZUREML_RUN_TOKEN_EXPIRY"],
        )

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


def _request_api(
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

    # print headers without disclosing token
    headers_output = {
        h: (headers[h] if h not in [AUTHORIZATION] else "*" * len(headers[h]))
        for h in headers
    }
    print(
        f"Sending request \n    to endpoint: {endpoint_url}"
        f"\n    with headers: {headers_output}"
        f"\n    and params: {request_params}"
    )
    response = session.post(
        endpoint_url,
        headers=headers,
        json=request_params,
        timeout=HTTP_REQUEST_TIMEOUT)
    time_taken = str(time.time() - time_start)
    print(f"Received response from endpoint: {response.__dict__}")
    print(f"Time taken to receive response: {time_taken}")
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


def _query_relevance_score(
    #turns: List[Tuple[str, str]],
    turns: List[Tuple[str, str]],
    session: requests.Session,
    endpoint_url: str,
    token_manager: _APITokenManager,
    # request_error_rate_threshold: float,
    model: str, temperature: float, top_p: float, num_samples: int,
    frequency_penalty: float, presence_penalty: float, max_tokens=3000, stop: str = None,
) -> List[int]:
    # if we count too many errors, we stop and raise an exception
    # error_count = 0
    # request_count = 0

    # request_count += 1

    # Copy request_data to avoid modifying the original dict.
    prompts = [RELEVANCE_ANNOTATION_TEMPLATE.replace("{input_samples", f"\n{json.dumps({'prompt': turn[0], 'completion': turn[1]}, indent=4)}") for turn in turns]
    # print("prompts:", prompts)
    ratings = []
    for prompt in prompts:
        request_data = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "n": num_samples,
            "max_tokens": max_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "messages": [{"role": "user", "content": prompt}]
        }

        if stop:
            request_data["stop"] = stop

        response = {}
        try:
            response, time_taken = _request_api(
                session=session,
                endpoint_url=endpoint_url,
                token_manager=token_manager,
                **request_data,
            )

            # Append time taken to the line
            response["response_time_sec"] = time_taken
            print(response["samples"][0])
            rating = json.loads(response["samples"][0])["rating"]
            # print("===response===")
            # print("rating=", rating, type(rating))
            ratings.append(rating)

        except Exception as e:  # noqa: B902
            response["finish_reason"] = ["error"]
            response["error"] = [str(e)]
            raise e
    
    return ratings


def _check_and_format_azure_endpoint_url(
    url_pattern, domain_pattern_re, domain, api_version, model
):
    domain = domain.strip()
    if domain.endswith('/'):
        domain = domain[:-1]

    if not re.match(domain_pattern_re, domain):
        raise RuntimeError(f"Invalid Azure endpoint domain URL: {domain}.")

    url = url_pattern.format(domain, model)

    if api_version:
        url += f"?api-version={api_version}"

    return url


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--production_dataset", type=str, required=True)
    parser.add_argument("--metric_names", type=str, required=True)
    parser.add_argument("--model_deployment_name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--stop", type=str, default=None)

    parser.add_argument("--groundedness_rating_threshold", type=int, default=4)
    parser.add_argument("--similarity_rating_threshold", type=int, default=4)
    parser.add_argument("--relevance_rating_threshold", type=int, default=4)
    parser.add_argument("--fluency_rating_threshold", type=int, default=4)
    parser.add_argument("--coherence_rating_threshold", type=int, default=4)

    parser.add_argument("--prompt_column_name", type=str, default=PROMPT)
    parser.add_argument("--completion_column_name", type=str, default=COMPLETION)
    parser.add_argument("--context_column_name", type=str, default=CONTEXT)
    parser.add_argument("--ground_truth_column_name", type=str, default=GROUND_TRUTH)

    parser.add_argument("--sample_rate", type=float, required=False, default=1.0)
    parser.add_argument(
        "--request_error_rate_threshold",
        type=float,
        default=0.5,
        help="Fail if the running error rate for the endpoint requests "
        "raises above this threshold.",
    )
    parser.add_argument("--api_call_retry_backoff_factor", type=int, default=4)
    parser.add_argument("--api_call_retry_max_count", type=int, default=10)
    parser.add_argument("--histogram", type=str, required=True)
    parser.add_argument("--samples_index", type=str, required=True)
    parser.add_argument("--groundedness_violations", type=str, required=True)
    parser.add_argument("--fluency_violations", type=str, required=True)
    parser.add_argument("--relevance_violations", type=str, required=True)
    parser.add_argument("--coherence_violations", type=str, required=True)
    parser.add_argument("--similarity_violations", type=str, required=True)

    parser.add_argument("--workspace_connection_arm_id", type=str, required=True)
    # args = parser.parse_args()
    args = parser.parse_args(args=[
        '--production_dataset', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/521f5fe6-857f-478c-b0cf-18338b21d4d2/joined_data/',
        '--metric_names', 'AcceptableRelevanceScorePerInstance,AggregatedRelevancePassRate',
        '--model_deployment_name', 'gpt-35-turbo-v0301',
        '--workspace_connection_arm_id', '/subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourceGroups/azureml-rag-ci/providers/Microsoft.MachineLearningServices/workspaces/azureml-rag-westus2/connections/azure_open_ai',
        '--prompt_column_name', 'question',
        '--completion_column_name', 'output', '--context_column_name', 'context',
        '--ground_truth_column_name', 'ground_truth', '--sample_rate', '0.1', '--groundedness_rating_threshold', '4',
        '--relevance_rating_threshold', '4', '--similarity_rating_threshold', '4', '--fluency_rating_threshold', '4',
        '--coherence_rating_threshold', '4', '--histogram',
        'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/histogram/',
        '--samples_index', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/samples_index/',
        '--groundedness_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/groundedness_violations/',
        '--fluency_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/fluency_violations/',
        '--similarity_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/similarity_violations/',
        '--coherence_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/coherence_violations/',
        '--relevance_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/relevance_violations/'
    ])

    request_args = {
        arg: getattr(args, arg) for arg in OPENAI_REQUEST_PARAMS if hasattr(args, arg)
    }
    request_args["model"] = args.model_deployment_name
    # request_args["n"] = args.num_samples

    token_manager = _WorkspaceConnectionTokenManager(
            connection_name=args.workspace_connection_arm_id,
            auth_header=API_KEY)
    azure_endpoint_domain_name = token_manager.get_endpoint_domain().replace("https://", "")
    azure_openai_api_version = token_manager.get_api_version()

    azure_endpoint_url = _check_and_format_azure_endpoint_url(
        AZURE_OPENAI_API_COMPLETION_URL_PATTERN,
        AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE,
        azure_endpoint_domain_name,
        azure_openai_api_version,
        args.model_deployment_name  # mdoel
    )
    endpoint_url = azure_endpoint_url

    print(f"Determined endpoint URL {endpoint_url}")

    httpClient = _HTTPClientWithRetry(
        n_retry=args.api_call_retry_max_count,
        backoff_factor=args.api_call_retry_backoff_factor,
    )

    with httpClient.client as session:
        ratings = _query_relevance_score(
            [
                (
                    "What's the highest mountain in the world?",
                    "Mount Everest is the highest mountain in the world. It is located between Nepal and Tibet, "
                    "an autonomous region of China. With an elevation of 29,032 feet (8,849 meters), it is considered "
                    "the tallest point on Earth"
                ),
                (
                    "Should I upgrade to windows11?",
                    "In general, a new prod is better than the old one."
                ),
                (
                    "should I upgrade to windows 11?",
                    "Whether to upgrade to Windows 11 depends on several factors. First, your hardware compatibility, as Windows 11 has specific system requirements. Second, your essential applications are compatible with Windows 11. Third, consider if the new features and interface of Windows 11 appeal to you. Lastly, support for Windows 10 will continue until October 14, 2025."
                )
            ],
            session, endpoint_url, token_manager,
            **request_args,
        )
    print(ratings)
    # calculate_flow_input_output_relevance_score()
    # make two cohorts by good and bad relevance
    # get question, indexed doc relevance from node_run_info.lookup.output[0/1/2].score for the 2 cohorts
    # use t-test to check if they have statistics significant difference
    # if yes:
    # 	generate update index action
    # else:
    # 	generate general action


if __name__ == "__main__":
    run()
