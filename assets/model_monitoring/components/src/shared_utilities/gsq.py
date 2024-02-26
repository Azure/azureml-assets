# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for the LLM-annotation based histogram component.

This file currently contains all the logic for the LLM-annotation based
histogram component.
The reason for putting everything into a single file is that the executors
don't have access to other files by default.
The functionality to simplify this code injection is still being developed.
"""

import logging
import os
import re
import tempfile
import traceback

import pandas as pd
import requests
from azure.ai.generative.evaluate import evaluate
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from pyspark.sql.types import IntegerType, StructField, StructType, StringType
from shared_utilities import io_utils
from shared_utilities.momo_exceptions import InvalidInputError

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

TEST_CONNECTION = "test_connection"

RATING = "rating"
PROMPT = "prompt"
COMPLETION = "completion"
CONTEXT = "context"
GROUND_TRUTH = "ground_truth"
CORRELATION_ID = "trace_id"


# ==================  HTTP Constants ==================
# Timeout per each request: 5min
HTTP_REQUEST_TIMEOUT = 300

# ================= Endpoint Constants =================
AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE = r"^(?=.{1,255}$)(?!-)[a-zA-Z0-9-]{1,63}(?<!-)(\.(?!-)[a-zA-Z0-9-]{1,63}(?<!-))*\.(inference\.ml|openai)\.azure\.com(/openai)?$"  # noqa: E501
AZURE_OPENAI_API_DEPLOYMENT_URL_PATTERN = "https://{}/openai/deployments/{}"

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
    "request_error_rate_threshold",
    "api_call_retry_backoff_factor",
    "api_call_retry_max_count",
    "model",
]

THRESHOLD_PARAMS = [
    "groundedness_rating_threshold",
    "similarity_rating_threshold",
    "relevance_rating_threshold",
    "fluency_rating_threshold",
    "coherence_rating_threshold",
]

# ---

CL_100K_BASE = "cl100k_base"
GPT_35_TURBO = "gpt-35-turbo"
GPT_35_TURBO_16K = "gpt-35-turbo-16k"
GPT_4 = "gpt-4"
GPT_4_32K = "gpt-4-32k"

# ---

MIN_RATING = 1
MAX_RATING = 5

GROUNDEDNESS = "Groundedness"
RELEVANCE = "Relevance"
FLUENCY = "Fluency"
COHERENCE = "Coherence"
SIMILARITY = "Similarity"

QAC_METRIC_NAMES = [
    GROUNDEDNESS,
    RELEVANCE,
]
QA_METRIC_NAMES = [
    FLUENCY,
    COHERENCE
]
ALL_METRIC_NAMES = [
    "AcceptableGroundednessScorePerInstance",
    "AggregatedGroundednessPassRate",
    "AcceptableCoherenceScorePerInstance",
    "AggregatedCoherencePassRate",
    "AcceptableFluencyScorePerInstance",
    "AggregatedFluencyPassRate",
    "AcceptableSimilarityScorePerInstance",
    "AggregatedSimilarityPassRate",
    "AcceptableRelevanceScorePerInstance",
    "AggregatedRelevancePassRate",
]
GPT_GROUNDEDNESS = "gpt_groundedness"
GPT_RELEVANCE = "gpt_relevance"
GPT_FLUENCY = "gpt_fluency"
GPT_COHERENCE = "gpt_coherence"
GPT_SIMILARITY = "gpt_similarity"

COMPACT_METRIC_NAME_TO_COLUMN = {
    GROUNDEDNESS: GPT_GROUNDEDNESS,
    RELEVANCE: GPT_RELEVANCE,
    FLUENCY: GPT_FLUENCY,
    COHERENCE: GPT_COHERENCE,
    SIMILARITY: GPT_SIMILARITY
}

COLUMN_TO_COMPACT_METRIC_NAME = {v: k for k, v in COMPACT_METRIC_NAME_TO_COLUMN.items()}

OUTPUT_SPLITTING_REGEX = r"[# ]*Task #*\d+:?"

AUTHORIZATION = "Authorization"
BEARER = "Bearer"
API_KEY = "api-key"
AZURE = "azure"
API_VERSION = "2023-07-01-preview"
METADATA_APIVERSION = "ApiVersion"
METADATA_DEPLOYMENTAPIVERSION = "DeploymentApiVersion"
METADATA_APITYPE = "ApiType"

NUMERICAL = "numerical"
COUNT = "count"
METRIC_NAME = "metric_name"
METRIC_VALUE = "metric_value"
GROUP = "group"
GROUP_DIMENSION = "group_dimension"
SAMPLES_NAME = "samples_name"
ASSET = "asset"
THRESHOLD = "threshold_value"
PRODUCTION_ROW_COUNT = "production_data"
REFERENCE_ROW_COUNT = "reference_data"


def _check_and_format_azure_endpoint_url(
    url_pattern, domain_pattern_re, domain, api_version, model
):
    domain = domain.strip()
    if domain.endswith('/'):
        domain = domain[:-1]

    if not re.match(domain_pattern_re, domain):
        err_msg = f"Invalid Azure endpoint domain URL: {domain}."
        err_msg += " The domain must be in the format of 'inference.ml.azure.com' or 'openai.azure.com'."
        raise InvalidInputError(err_msg)

    url = url_pattern.format(domain, model)

    if api_version:
        url += f"?api-version={api_version}"

    return url


class _WorkspaceConnectionTokenManager(object):
    def __init__(
        self,
        *,
        connection_name,
        auth_header,
        **kwargs,
    ):
        self.credential = self.get_aad_credential()
        self.token = None
        self.auth_header = auth_header

        try:
            from azureml.dataprep.api._aml_auth._azureml_token_authentication import AzureMLTokenAuthentication
            from azure.ai.ml import MLClient
            from azure.ai.ml.entities import WorkspaceConnection

            self._credential = AzureMLTokenAuthentication._initialize_aml_token_auth()

            uri_match = re.match(r"/subscriptions/(.*)/resourceGroups/(.*)/providers/Microsoft.MachineLearningServices/workspaces/(.*)/connections/(.*)",  # noqa: E501
                                 connection_name, flags=re.IGNORECASE)

            subscription_id = uri_match.group(1)
            resource_group_name = uri_match.group(2)
            workspace_name = uri_match.group(3)
            ml_client = MLClient(
                credential=self._credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group_name,
                workspace_name=workspace_name
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
                api_version = API_VERSION
                if hasattr(connection.metadata, METADATA_APIVERSION):
                    api_version = connection.metadata[METADATA_APIVERSION]
                # this was renamed in latest ml_client
                if hasattr(connection.metadata, METADATA_DEPLOYMENTAPIVERSION):
                    api_version = connection.metadata[METADATA_DEPLOYMENTAPIVERSION]
                # api version
                self.api_version = api_version
                # base_url
                self.domain_name = connection.target
                # api_key
                self.token = connection.credentials["key"]
                self.api_type = None
                if hasattr(connection.metadata, METADATA_APITYPE):
                    self.api_type = connection.metadata[METADATA_APITYPE]
            else:
                raise Exception("Unable to retrieve the token to establish a Workspace Connection")
        except Exception:
            tb = traceback.format_exc()
            raise Exception(f"Error encountered while attempting to authentication token: {tb}")

    def get_aad_credential(self):
        return AzureMLOnBehalfOfCredential(
            AZUREML_SYNAPSE_CLUSTER_IDENTIFIER=os.environ[
                "AZUREML_SYNAPSE_CLUSTER_IDENTIFIER"
            ],
            AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT=os.environ[
                "AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT"
            ],
            AZUREML_RUN_ID=os.environ["AZUREML_RUN_ID"],
            AZUREML_RUN_TOKEN_EXPIRY=os.environ["AZUREML_RUN_TOKEN_EXPIRY"],
        )

    def get_api_version(self):
        return self.api_version

    def get_endpoint_domain(self):
        return self.domain_name

    def get_token(self):
        return self.token


def _get_model_type(token_manager, get_model_endpoint):
    try:
        headers = {
            "Content-Type": "application/json",
            "api-key": token_manager.get_token()
        }
        response = requests.get(url=get_model_endpoint, headers=headers, timeout=HTTP_REQUEST_TIMEOUT)
        if response.status_code == 200:
            response_data = response.json()
            model_type = response_data["model"]
        else:
            raise Exception(
                "Received unexpected HTTP status: "
                f"{response.status_code} {response.text}"
            )
    except Exception:
        raise Exception("Error encountered while attempting to get model type")
    return model_type


def get_compact_metric_name(metric_name):
    """Get the compact metric name from the full metric name."""
    return metric_name.replace(" ", "").title()


def process_metric_names(metric_names):
    """Process metric names, remove whitespace and prune."""
    input_metric_names = [m.strip() for m in metric_names.split(",")]

    if not (set(input_metric_names) <= set(ALL_METRIC_NAMES)):
        raise InvalidInputError(
            f"metric_names must be a comma-separated list of metric names "
            f"and a subset of {ALL_METRIC_NAMES}, got {metric_names}."
        )

    # remove all but groundedness/fluency/coherence/relevance/similarity from metric names and
    # remove duplicates
    pruned_metric_names = [re.sub(r'^(.*?)(Groundedness|Fluency|Coherence|Relevance|Similarity)(.*?)$', r'\2', m) for
                           m in input_metric_names]
    metric_names = list(set(pruned_metric_names))
    return metric_names


def get_request_arg_or_default(arg, request_args):
    """Get request arg or default."""
    return request_args[arg] if arg in request_args else None


def validate_parameters(request_args, sample_rate):
    """Validate input parameters."""
    temperature = get_request_arg_or_default("temperature", request_args)
    top_p = get_request_arg_or_default("top_p", request_args)
    num_samples = get_request_arg_or_default("num_samples", request_args)
    frequency_penalty = get_request_arg_or_default("frequency_penalty", request_args)
    presence_penalty = get_request_arg_or_default("presence_penalty", request_args)
    if temperature is not None and (temperature < 0.0 or temperature > 2.0):
        raise InvalidInputError(f"temperature must be between 0.0 and 2.0, inclusive; "
                                f"got {temperature}.")
    if top_p is not None and (top_p < 0.0 or top_p > 1.0):
        raise InvalidInputError(
            f"top_p must be between 0.0 and 1.0, inclusive; got {top_p}.")
    if num_samples is not None and num_samples <= 0:
        # TODO support multiple returned annotations
        raise InvalidInputError(f"num_samples must be 1, got {num_samples}.")
    if frequency_penalty is not None and (frequency_penalty < -2.0 or frequency_penalty > 2.0):
        raise InvalidInputError(
            "frequency_penalty must be between -2.0 and 2.0, inclusive; "
            f"got {frequency_penalty}."
        )
    if presence_penalty is not None and (presence_penalty < -2.0 or presence_penalty > 2.0):
        raise InvalidInputError(
            f"presence_penalty must be between -2.0 and 2.0, inclusive; "
            f"got {presence_penalty}."
        )

    if sample_rate <= 0.0 or sample_rate > 1.0:
        raise InvalidInputError(
            f"sample_rate must be larger than 0.0 and at most 1.0, "
            f"got {sample_rate}.")


def apply_annotation(
    *,
    metric_names,
    production_df,
    model_deployment_name,
    workspace_connection_arm_id,
    num_samples,
    sample_rate,
    request_args,
    prompt_column_name,
    completion_column_name,
    context_column_name,
    ground_truth_column_name
):
    """Apply annotation to all samples in the production_dataset."""
    metric_names = process_metric_names(metric_names)
    validate_parameters(request_args, sample_rate)

    if "chat_history" in [prompt_column_name, completion_column_name, context_column_name, ground_truth_column_name]:
        raise NotImplementedError("chat_history column is not currently supported and cannot be used as specified "
                                  "column. ")
    # Ensure input data has the correct columns given the metrics
    # Question, answer required for coherence and fluency
    qa_required = len(list(set(QA_METRIC_NAMES).intersection(
        set(metric_names))))
    for col_name in [prompt_column_name, completion_column_name]:
        if col_name not in production_df.columns and qa_required:
            raise InvalidInputError(f"production_dataset must have column: {col_name}")
    # Question, answer, context required for relevance and groundedness
    qac_required = len(list(set(QAC_METRIC_NAMES).intersection(
        set(metric_names))))
    if qac_required and context_column_name not in production_df.columns:
        raise InvalidInputError(f"production_dataset must have column: {context_column_name}")
    # Question, answer, ground-truth required for similarity
    if SIMILARITY in metric_names and ground_truth_column_name not in production_df.columns:
        raise InvalidInputError(f"production_dataset must have column: {ground_truth_column_name}")

    column_names = [prompt_column_name, completion_column_name, context_column_name, ground_truth_column_name]
    if len(column_names) != len(set(column_names)):
        raise InvalidInputError(
            "Detected duplicate specified columns. Column name input cannot be the same. Please ensure "
            f"that the column input specified is unique.\nReceived prompt_column_name: "
            f"{prompt_column_name}\ncompletion_column_name: {completion_column_name}\n"
            f"context_column_name: {context_column_name}\nground_truth_column_name: "
            f"{ground_truth_column_name}")

    columns_to_select = [prompt_column_name, completion_column_name]
    renamed_columns = [PROMPT, COMPLETION]
    if context_column_name in production_df.columns:
        columns_to_select.append(context_column_name)
        renamed_columns.append(CONTEXT)
    if ground_truth_column_name in production_df.columns:
        columns_to_select.append(ground_truth_column_name)
        renamed_columns.append(GROUND_TRUTH)
    has_correlation_id = CORRELATION_ID in production_df.columns
    if has_correlation_id:
        columns_to_select.append(CORRELATION_ID)
        renamed_columns.append(CORRELATION_ID)

    # select the relevant column names
    production_df = production_df.select(columns_to_select)

    # rename columns to prompt, completion, context, ground truth to match metaprompt data
    production_df = (production_df.withColumnRenamed(prompt_column_name, PROMPT)
                     .withColumnRenamed(completion_column_name, COMPLETION)
                     .withColumnRenamed(context_column_name, CONTEXT)
                     .withColumnRenamed(ground_truth_column_name, GROUND_TRUTH))
    production_df = production_df.select(renamed_columns)
    # Sampling
    production_df_sampled = production_df.sample(withReplacement=False, fraction=sample_rate)
    if production_df_sampled.count() == 0:
        print("Not enough data resulting from sample_rate and production dataset. "
              "Using first five rows of production dataset instead. To use custom sample_rate with this dataset, "
              "try increasing sample_rate value.")
        # Default to 5
        production_df_sampled = production_df.limit(5)

    production_df = production_df_sampled

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
    is_test_connection = False
    if workspace_connection_arm_id == TEST_CONNECTION:
        # Used for testing component e2e without consuming OpenAI endpoint
        endpoint_domain_name = TEST_CONNECTION
        api_version = API_VERSION
        is_test_connection = True
        token_manager = None
        model_type = GPT_4
    else:
        try:
            # Define authorization token manager
            token_manager_class = _WorkspaceConnectionTokenManager
            token_manager = token_manager_class(
                connection_name=workspace_connection_arm_id,
                auth_header=API_KEY
            )
        except Exception as e:
            print(f"Unable to process request: {e}")
            return

        endpoint_domain_name = token_manager.get_endpoint_domain().replace("https://", "")
        api_version = token_manager.get_api_version()
        api_key = token_manager.get_token()
        api_base = token_manager.get_endpoint_domain()

        print(
            "Created token manager for auth type "
            f"managed identity using auth header {API_KEY}."
        )
        # use fixed API version since newer versions aren't supported
        get_model_endpoint = _check_and_format_azure_endpoint_url(
            AZURE_OPENAI_API_DEPLOYMENT_URL_PATTERN,
            AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE,
            endpoint_domain_name, "2022-12-01",
            model_deployment_name)
        model_type = _get_model_type(token_manager, get_model_endpoint)

    metrics_list = []
    for metric_name in metric_names:
        metric_name_compact = get_compact_metric_name(metric_name)
        column_name = COMPACT_METRIC_NAME_TO_COLUMN[metric_name_compact]
        metrics_list.append(column_name)
    has_context = CONTEXT in production_df.columns
    has_ground_truth = GROUND_TRUTH in production_df.columns

    def annotate_batch(iterator):
        for batch in iterator:
            # add environment variables on executors
            for env_var_key, env_var_value in driver_env_vars.items():
                os.environ[env_var_key] = env_var_value

            rows = []
            correlation_ids = None
            if has_correlation_id:
                correlation_ids = batch[CORRELATION_ID]
            for index, row in batch.iterrows():
                qca = {PROMPT: row[PROMPT], COMPLETION: row[COMPLETION]}
                if has_context:
                    qca[CONTEXT] = row[CONTEXT]
                if has_ground_truth:
                    qca[GROUND_TRUTH] = row[GROUND_TRUTH]
                rows.append(qca)

            output_dir = tempfile.TemporaryDirectory()
            evaluate(
                evaluation_name="gsq-evaluation",
                data=rows,
                task_type="qa",
                data_mapping={
                    "question": PROMPT,
                    "context": CONTEXT,
                    "answer": COMPLETION,
                    "ground_truth": GROUND_TRUTH
                },
                model_config={
                    "api_version": api_version,
                    "api_base": api_base,
                    "api_type": AZURE,
                    "api_key": api_key,
                    "deployment_id": model_type
                },
                metrics_list=metrics_list,
                output_path=output_dir.name
            )
            tabular_result = pd.read_json(os.path.join(output_dir.name, "eval_results.jsonl"), lines=True)
            if correlation_ids is not None:
                tabular_result[CORRELATION_ID] = correlation_ids
            # rename metric columns
            for column_name in metrics_list:
                # set failures to -1
                tabular_result[column_name] = pd.to_numeric(tabular_result[column_name], errors='coerce')
                tabular_result[column_name].fillna(-1, inplace=True)
                tabular_result.rename(
                    columns={column_name: COLUMN_TO_COMPACT_METRIC_NAME[column_name]},
                    inplace=True)
            yield tabular_result

    # used for testing without using openai connection
    def mock_metrics_batch(iterator):
        for batch in iterator:
            rows = []
            correlation_ids = None
            if has_correlation_id:
                correlation_ids = batch[CORRELATION_ID]
            for index, row in batch.iterrows():
                qca = {PROMPT: row[PROMPT], COMPLETION: row[COMPLETION]}
                if has_context:
                    qca[CONTEXT] = row[CONTEXT]
                if has_ground_truth:
                    qca[GROUND_TRUTH] = row[GROUND_TRUTH]
                for metric_name in metric_names:
                    metric_name_compact = get_compact_metric_name(metric_name)
                    qca[metric_name_compact] = 1
                rows.append(qca)
            tabular_result = pd.DataFrame(rows)
            if correlation_ids is not None:
                tabular_result[CORRELATION_ID] = correlation_ids
            yield tabular_result

    schema_fields = [
        StructField(PROMPT, StringType(), True)
    ]
    if has_context:
        schema_fields.append(StructField(CONTEXT, StringType(), True))
    schema_fields.append(StructField(COMPLETION, StringType(), True))
    if has_ground_truth:
        schema_fields.append(StructField(GROUND_TRUTH, StringType(), True))
    if has_correlation_id:
        schema_fields.append(StructField(CORRELATION_ID, StringType(), True))
    for metric_name in metric_names:
        metric_name_compact = get_compact_metric_name(metric_name)
        schema_fields.append(StructField(metric_name_compact, IntegerType(), True))
    schema = StructType(schema_fields)
    if is_test_connection:
        annotations_df = production_df.mapInPandas(
            mock_metrics_batch,
            schema=schema,
        ).cache()
    else:
        annotations_df = production_df.mapInPandas(
            annotate_batch,
            schema=schema,
        ).cache()

    print("showing annotations dataframe: ")
    annotations_df.show()
    return annotations_df
