# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for the LLM-annotation based histogram component.

This file currently contains all the logic for the LLM-annotation based
histogram component.
The reason for putting everything into a single file is that the executors
don't have access to other files by default.
The functionality to simplify this code injection is still being developed.
"""

import argparse
import logging
import os
import re
import tempfile
import mlflow
import socket
import json
import uuid

import pandas as pd
from azure.ai.evaluation._model_configurations import AzureOpenAIModelConfiguration
from azure.ai.evaluation import (
    evaluate,
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    SimilarityEvaluator)
from shared_utilities.llm_utils import _WorkspaceConnectionTokenManager
from pyspark.sql.types import IntegerType, StructField, StructType, StringType
from pyspark.sql.functions import col
from shared_utilities.io_utils import (
    try_read_mltable_in_spark_with_error,
    save_spark_df_as_mltable,
    init_spark,
)
from shared_utilities.momo_exceptions import InvalidInputError

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

TEST_CONNECTION = "test_connection"

RATING = "rating"
PROMPT = "prompt"
COMPLETION = "completion"
CONTEXT = "context"
GROUND_TRUTH = "ground_truth"
QUERY = "query"
RESPONSE = "response"
COLUMN_MAPPING = "column_mapping"
CORRELATION_ID = "correlationid"
TRACE_ID = "trace_id"
ROOT_SPAN = "root_span"

PASSTHROUGH_COLUMNS = [CORRELATION_ID, TRACE_ID, ROOT_SPAN]

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

EVALUATOR_NAME_TO_CLASS = {
    GPT_GROUNDEDNESS: GroundednessEvaluator,
    GPT_RELEVANCE: RelevanceEvaluator,
    GPT_FLUENCY: FluencyEvaluator,
    GPT_COHERENCE: CoherenceEvaluator,
    GPT_SIMILARITY: SimilarityEvaluator
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

DEFAULT_PROMPTFLOW_PATH = "/home/trusted-service-user/.promptflow/"


def get_compact_metric_name(metric_name):
    """Get the compact metric name from the full metric name."""
    return metric_name.replace(" ", "").title()


def run():
    """Compute metrics."""
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
    parser.add_argument("--evaluation", type=str, required=True)
    args = parser.parse_args()

    request_args = {
        arg: getattr(args, arg) for arg in OPENAI_REQUEST_PARAMS if hasattr(args, arg)
    }
    endpoint_args = {
        arg: getattr(args, arg) for arg in ENDPOINT_PARAMS if hasattr(args, arg)
    }
    threshold_args = {
        arg: getattr(args, arg) for arg in THRESHOLD_PARAMS if hasattr(args, arg)
    }
    # add model to both request and endpoint args
    # The arg name is longer to be as explicit as possible.
    request_args["model"] = args.model_deployment_name
    endpoint_args["model"] = args.model_deployment_name

    # TODO add validation for threshold args!!
    print(f"Running with args: {args}")

    violations = {
        "groundedness": args.groundedness_violations,
        "relevance": args.relevance_violations,
        "fluency": args.fluency_violations,
        "similarity": args.similarity_violations,
        "coherence": args.coherence_violations,
    }

    apply_annotation(
        metric_names=args.metric_names,
        production_dataset=args.production_dataset,
        histogram=args.histogram,
        samples_index=args.samples_index,
        model_deployment_name=args.model_deployment_name,
        workspace_connection_arm_id=args.workspace_connection_arm_id,
        num_samples=args.num_samples,
        sample_rate=args.sample_rate,
        request_args=request_args,
        endpoint_args=endpoint_args,
        threshold_args=threshold_args,
        prompt_column_name=args.prompt_column_name,
        completion_column_name=args.completion_column_name,
        context_column_name=args.context_column_name,
        ground_truth_column_name=args.ground_truth_column_name,
        violations=violations,
        evaluation=args.evaluation
    )


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


def get_passthrough_cols(df):
    """Get passthrough columns as a dictionary from name to values."""
    passthrough_cols = {}
    for passthrough_column in PASSTHROUGH_COLUMNS:
        if passthrough_column in df.columns:
            passthrough_cols[passthrough_column] = df[passthrough_column]
    return passthrough_cols


def format_data_column(column):
    """Format data column in promptflow-evals config format."""
    return "${data." + column + "}"


def apply_annotation(
    *,
    metric_names,
    production_dataset,
    histogram,
    model_deployment_name,
    workspace_connection_arm_id,
    num_samples,
    sample_rate,
    request_args,
    endpoint_args,
    threshold_args,
    prompt_column_name,
    completion_column_name,
    context_column_name,
    ground_truth_column_name,
    samples_index,
    violations,
    evaluation
):
    """Apply annotation to all samples in the production_dataset."""
    metric_names = process_metric_names(metric_names)
    validate_parameters(request_args, sample_rate)

    production_df = try_read_mltable_in_spark_with_error(production_dataset, "production_dataset")
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
    for passthrough_column in PASSTHROUGH_COLUMNS:
        if passthrough_column in production_df.columns:
            columns_to_select.append(passthrough_column)
            renamed_columns.append(passthrough_column)

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
    row_count = production_df.count()

    spark = init_spark()
    spark_conf = spark.sparkContext.getConf()
    spark_conf_vars = {
        "AZUREML_SYNAPSE_CLUSTER_IDENTIFIER": "spark.synapse.clusteridentifier",  # noqa: E501
        "AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT": "spark.tokenServiceEndpoint",
        "TID": "spark.synapse.workspace.tenantId"
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
            "AZUREML_RUN_TOKEN",
            "AZUREML_RUN_TOKEN_EXPIRY",
            "AZUREML_OBO_SERVICE_ENDPOINT",
            "AZUREML_OBO_CANARY_TOKEN",
            "AZUREML_ARM_SUBSCRIPTION",
            "AZUREML_ARM_RESOURCEGROUP",
            "AZUREML_ARM_WORKSPACE_NAME",
            "AZUREML_ARM_PROJECT_NAME",
            "MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING",
            "MLFLOW_EXPERIMENT_ID",
            "MLFLOW_EXPERIMENT_NAME",
            "MLFLOW_RUN_ID",
            "MLFLOW_TRACKING_TOKEN",
            "MLFLOW_TRACKING_URI",
            "OID",
            "TID"
        ]
    }
    is_test_connection = False
    if workspace_connection_arm_id == TEST_CONNECTION:
        # Used for testing component e2e without consuming OpenAI endpoint
        api_version = API_VERSION
        is_test_connection = True
        token_manager = None
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

        api_version = token_manager.get_api_version()
        api_key = token_manager.get_token()
        api_base = token_manager.get_endpoint_domain()

        print(
            "Created token manager for auth type "
            f"managed identity using auth header {API_KEY}."
        )

    all_metrics_pdf = None
    samples_index_rows = []
    metrics_list = []
    compact_metric_names = []
    for metric_name in metric_names:
        metric_name_compact = get_compact_metric_name(metric_name)
        compact_metric_names.append(metric_name_compact)
        column_name = COMPACT_METRIC_NAME_TO_COLUMN[metric_name_compact]
        metrics_list.append(column_name)
    has_context = CONTEXT in production_df.columns
    has_ground_truth = GROUND_TRUTH in production_df.columns

    # get tracking uri
    tracking_uri = mlflow.get_tracking_uri()
    run_id = os.environ.get("AZUREML_RUN_ID")

    def annotate_batch(iterator):
        for batch in iterator:
            # add environment variables on executors
            for env_var_key, env_var_value in driver_env_vars.items():
                os.environ[env_var_key] = env_var_value
            rows = []
            input_columns = [PROMPT, COMPLETION]
            if has_context:
                input_columns.append(CONTEXT)
            if has_ground_truth:
                input_columns.append(GROUND_TRUTH)
            passthrough_cols = get_passthrough_cols(batch)
            for index, row in batch.iterrows():
                qca = {QUERY: row[PROMPT], RESPONSE: row[COMPLETION]}
                if has_context:
                    qca[CONTEXT] = row[CONTEXT]
                if has_ground_truth:
                    qca[GROUND_TRUTH] = row[GROUND_TRUTH]
                rows.append(qca)

            if not os.path.exists(DEFAULT_PROMPTFLOW_PATH):
                os.makedirs(DEFAULT_PROMPTFLOW_PATH, exist_ok=True)
            mlflow.set_tracking_uri(tracking_uri)

            input_dir = tempfile.TemporaryDirectory()
            evaluation_name = "gsq-evaluation"
            run_name = evaluation_name + "-child-run"
            model_config = AzureOpenAIModelConfiguration(
                azure_endpoint=api_base,
                api_key=api_key,
                azure_deployment=model_deployment_name,
                api_version=api_version,
            )
            evaluators = {}
            evaluator_config = {}
            for metric_name in metrics_list:
                evaluator = EVALUATOR_NAME_TO_CLASS[metric_name]
                metric_name_compact = COLUMN_TO_COMPACT_METRIC_NAME[
                    metric_name].lower()
                evaluators[metric_name_compact] = evaluator(
                    model_config=model_config)
                evaluator_config[metric_name_compact] = {
                    COLUMN_MAPPING: {
                        RESPONSE: format_data_column(RESPONSE),
                        QUERY: format_data_column(QUERY)
                    }
                }
                config = evaluator_config[metric_name_compact]
                if has_context:
                    config[COLUMN_MAPPING][CONTEXT] = format_data_column(CONTEXT)
                if has_ground_truth:
                    config[COLUMN_MAPPING][GROUND_TRUTH] = format_data_column(GROUND_TRUTH)
            # write rows to jsonl file
            input_file_name = "eval_input_" + str(uuid.uuid4()) + ".jsonl"
            input_file_path = os.path.join(input_dir.name, input_file_name)
            with open(input_file_path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')
            # get existing run
            with mlflow.start_run():
                # create child run
                with mlflow.start_run(nested=mlflow.active_run(), run_name=run_name) as run:
                    tabular_result = evaluate(
                        evaluation_name=evaluation_name,
                        data=input_file_path,
                        evaluators=evaluators,
                        evaluator_config=evaluator_config
                    )
                    child_run_id = run.info.run_id
                    # add promptflow debug logs to run
                    hostname = socket.gethostname()
                    artifact_path = f"worker_promptflow/{hostname}"
                    client = mlflow.tracking.MlflowClient()
                    client.log_artifacts(
                        child_run_id, DEFAULT_PROMPTFLOW_PATH, artifact_path=artifact_path)
            # convert rows from result to pandas dataframe
            tabular_result = pd.DataFrame(tabular_result["rows"])
            for passthrough_column, passthrough_values in passthrough_cols.items():
                tabular_result[passthrough_column] = passthrough_values
            # rename metric columns
            try:
                for column_name in metrics_list:
                    # set failures to -1
                    metric_name_compact = COLUMN_TO_COMPACT_METRIC_NAME[column_name]
                    # output column names follow schema like "outputs.coherence.gpt_coherence"
                    result_name = ("outputs." + metric_name_compact + "." + column_name).lower()
                    tabular_result[result_name] = pd.to_numeric(tabular_result[result_name], errors='coerce')
                    tabular_result[result_name].fillna(-1, inplace=True)
                    tabular_result.rename(columns={result_name: metric_name_compact}, inplace=True)
                for column_name in input_columns:
                    # input column names follow schema like "inputs.context"
                    if column_name == PROMPT:
                        result_name = "inputs." + QUERY
                    elif column_name == COMPLETION:
                        result_name = "inputs." + RESPONSE
                    else:
                        result_name = "inputs." + column_name
                    tabular_result.rename(columns={result_name: column_name}, inplace=True)
            except KeyError as e:
                # raise new user error with more context
                raise InvalidInputError(
                    f"Unable to retrieve and rename {column_name}. "
                    f"Usually this indicates an invalid configuration or connection. "
                    f"Please check the model_deployment_name and workspace_connection_arm_id. "
                    f"For more detailed error information please see the promptflow "
                    f"debug folder in the child run with run id {child_run_id}."
                ) from e
            yield tabular_result

    # used for testing without using openai connection
    def mock_metrics_batch(iterator):
        for batch in iterator:
            rows = []
            passthrough_cols = get_passthrough_cols(batch)
            for index, row in batch.iterrows():
                qca = {PROMPT: row[PROMPT], COMPLETION: row[COMPLETION]}
                if has_context:
                    qca[CONTEXT] = row[CONTEXT]
                if has_ground_truth:
                    qca[GROUND_TRUTH] = row[GROUND_TRUTH]
                for metric_name_compact in compact_metric_names:
                    qca[metric_name_compact] = 1
                rows.append(qca)
            tabular_result = pd.DataFrame(rows)
            for passthrough_column, passthrough_values in passthrough_cols.items():
                tabular_result[passthrough_column] = passthrough_values
            yield tabular_result

    schema_fields = [
        StructField(PROMPT, StringType(), True)
    ]
    if has_context:
        schema_fields.append(StructField(CONTEXT, StringType(), True))
    schema_fields.append(StructField(COMPLETION, StringType(), True))
    if has_ground_truth:
        schema_fields.append(StructField(GROUND_TRUTH, StringType(), True))
    for passthrough_column in PASSTHROUGH_COLUMNS:
        if passthrough_column in production_df.columns:
            schema_fields.append(StructField(passthrough_column, StringType(), True))
    for metric_name_compact in compact_metric_names:
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

    for metric_name_compact in compact_metric_names:
        # Run inference over input dataset
        print(f"Begin {metric_name_compact} processing.")
        # Get rating counts
        filtered_annotations_df = annotations_df.select(
            metric_name_compact).filter(col(metric_name_compact) != -1)
        metrics_df = filtered_annotations_df.groupBy(metric_name_compact).count()
        metrics_df.show()
        print("Finished annotating answers.")
        metrics_pdf = metrics_df.withColumnRenamed(metric_name_compact, RATING).select("*").toPandas()
        print(metrics_pdf)
        ratings = metrics_pdf.rating.to_list()
        missing_ratings = set(range(MIN_RATING, MAX_RATING + 1)) - set(ratings)
        for r in missing_ratings:
            metrics_pdf.loc[len(metrics_pdf)] = {RATING: r, COUNT: 0}
        metrics_pdf[RATING] = metrics_pdf[RATING].map(lambda r: str(r))
        # add metric_name, metric_value, group, and threshold values
        metrics_pdf.rename(columns={RATING: GROUP, COUNT: METRIC_VALUE, }, inplace=True)
        metrics_pdf[METRIC_NAME] = f"Acceptable{metric_name_compact}ScorePerInstance"
        metric_threshold_value = str(threshold_args[f"{metric_name_compact.lower()}_rating_threshold"])
        metrics_pdf[THRESHOLD] = metric_threshold_value
        print(metrics_pdf)

        # create violations table if there are violations
        violation_columns = renamed_columns.copy()
        violation_columns.append(metric_name_compact)
        violations_df = annotations_df.select(violation_columns).filter(
                (col(metric_name_compact) < metric_threshold_value) & (col(metric_name_compact) != -1))
        if violations_df.count() > 0:
            # rename columns back to original names
            violations_df = (violations_df.withColumnRenamed(PROMPT, prompt_column_name)
                             .withColumnRenamed(COMPLETION, completion_column_name)
                             .withColumnRenamed(CONTEXT, context_column_name)
                             .withColumnRenamed(GROUND_TRUTH, ground_truth_column_name))
            save_spark_df_as_mltable(violations_df, violations[metric_name_compact.lower()])
            samples_index_rows.append({METRIC_NAME: f"Acceptable{metric_name_compact}ScorePerInstance",
                                       GROUP: "",
                                       GROUP_DIMENSION: "",
                                       SAMPLES_NAME: "Violations",
                                       ASSET: f"azureml_{run_id}_output_data_{metric_name_compact.lower()}_violations:1"})  # noqa: E501

        if all_metrics_pdf is None:
            all_metrics_pdf = metrics_pdf
        else:
            all_metrics_pdf = pd.concat([all_metrics_pdf, metrics_pdf])

    print(f"Adding {PRODUCTION_ROW_COUNT} and {REFERENCE_ROW_COUNT}.")
    for row_count_name in [PRODUCTION_ROW_COUNT, REFERENCE_ROW_COUNT]:
        row = {
            METRIC_NAME: "RowCount",
            METRIC_VALUE: row_count,
            GROUP: row_count_name,
            THRESHOLD: ""
        }
        all_metrics_pdf.loc[len(all_metrics_pdf)] = row
    print("Finished calculating metrics based on annotations.")

    metadata_schema = StructType(
        [
            StructField(METRIC_NAME, StringType(), True),
            StructField(GROUP, StringType(), True),
            StructField(GROUP_DIMENSION, StringType(), True),
            StructField(SAMPLES_NAME, StringType(), True),
            StructField(ASSET, StringType(), True),
        ]
    )

    samples_index_rows.append({METRIC_NAME: "Evaluation",
                               GROUP: "",
                               GROUP_DIMENSION: "",
                               SAMPLES_NAME: "Evaluation",
                               ASSET: f"azureml_{run_id}_output_data_evaluation:1"})
    # Create a new DataFrame for the samples index data
    samples_df = spark.createDataFrame(samples_index_rows, metadata_schema)

    # Save the samples and annotations dataframes as output
    save_spark_df_as_mltable(annotations_df, evaluation)
    save_spark_df_as_mltable(samples_df, samples_index)

    # temporary workaround for pandas>2.0 until pyspark upgraded to 3.4.1, see issue:
    # https://stackoverflow.com/questions/76404811/attributeerror-dataframe-object-has-no-attribute-iteritems
    pd.DataFrame.iteritems = pd.DataFrame.items
    save_spark_df_as_mltable(
        spark.createDataFrame(all_metrics_pdf),
        histogram)


if __name__ == "__main__":
    run()
