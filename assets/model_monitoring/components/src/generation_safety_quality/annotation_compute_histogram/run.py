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

import json5
import pandas as pd
import requests
import tiktoken
from azure.ai.ml.identity import CredentialUnavailableError
from azure.ai.ml.identity._internal import _scopes_to_resource
from azure.core.credentials import AccessToken, TokenCredential
from pyspark.sql import DataFrame, Window
from pyspark.sql.types import IntegerType, StructField, StructType, StringType
from pyspark.sql.functions import col, row_number, monotonically_increasing_id
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from shared_utilities import io_utils

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

DEFAULT_INDENT = 2

RATING = "rating"
INDEX = "index"
LABEL_KEYS = [RATING]
PROMPT = "prompt"
COMPLETION = "completion"
CONTEXT = "context"
GROUND_TRUTH = "ground_truth"


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
AZURE_OPENAI_API_COMPLETION_URL_PATTERN = "https://{}/openai/deployments/{}/chat/completions"
LISTSECRETS_API_PATTERN = "https://management.azure.com{}/listsecrets?api-version=2023-06-01-preview"
AZURE_OPENAI_API_DEPLOYMENT_URL_PATTERN = "https://{}/openai/deployments/{}"

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
BASE_MAX_TOKENS = 1000
BASE_MAX_INPUTS = 10
BASE_MAX_PROMPT_TOKENS = 3000
MODEL_TYPE_FACTOR = {
    GPT_35_TURBO: 1,
    GPT_4: 2,
    GPT_35_TURBO_16K: 4,
    GPT_4_32K: 8
}

# ---

MIN_RATING = 1
MAX_RATING = 5

GROUNDING_ANNOTATION_TEMPLATE = "\n\n".join(
    [
        "System:",
        f"Given the {CONTEXT} and {PROMPT}, score the {COMPLETION} between {MIN_RATING} to {MAX_RATING} {RATING}, \
            where {MIN_RATING} star means \"inconsistency\" and {MAX_RATING} {RATING} means \"perfect consistency\". \
                Note that consistency measures whether the facts in the answer are consistent with the facts in the \
                    context. Consider whether the answer does reproduce facts accurately and does not make up \
                        untrue information. Answer only as an integer only, between 1 and 5.",
        "## Example Task #0",
        json.dumps({
            CONTEXT: "The Oscars are a popular award ceremony.",
            PROMPT: "Did Lord of the Rings win any Oscars?",
            COMPLETION: "Lord of the Rings won 10 Oscars.",
        }),
        "A good example response would be:",
        "## Example Task #0:",
        json.dumps({
            RATING: MIN_RATING,
        }),
        "User:",
        "{input_samples}",
        "Reminder: The return values for each task should be dictionaries with the key 'rating'. The value of \
            'rating' should be an integer between 1 and 5."
    ]
)

RELEVANCE_ANNOTATION_TEMPLATE = "\n\n".join(
    [
        "System:",
        f"You are an AI assistant. You will be given the definition of an evaluation metric for assessing the \
            quality of an {COMPLETION} in a question-answering task. Your job is to compute an accurate evaluation \
                score using the provided evaluation metric.",
        f"Relevance measures how well the {COMPLETION} addresses the main aspects of the {PROMPT}, based on the \
            {CONTEXT}. Consider whether all and only the important aspects are contained in the {COMPLETION} when \
                evaluating relevance. Given the {CONTEXT} and {PROMPT}, score the relevance of the {COMPLETION} \
                    between {MIN_RATING} to {MAX_RATING} using the following {RATING} scale:",
        f"{RATING} 1: the answer completely lacks relevance",
        f"{RATING} 2: the answer mostly lacks relevance",
        f"{RATING} 3: the answer is partially relevant",
        f"{RATING} 4: the answer is mostly relevant",
        f"{RATING} 5: the answer has perfect relevance",
        f"The score should be integer only, between {MIN_RATING} and {MAX_RATING}.",
        "## Example Task #0",
        json.dumps({
            CONTEXT: "Marie Curie was a Polish-born physicist and chemist who pioneered research on radioactivity \
                and was the first woman to win a Nobel Prize.",
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
            CONTEXT: "The Beatles were an English rock band formed in Liverpool in 1960, and they are widely \
                regarded as the most influential music band in history.",
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
            CONTEXT: "The recent Mars rover, Perseverance, was launched in 2020 with the main goal of searching \
                for signs of ancient life on Mars. The rover also carries an experiment called MOXIE, which aims \
                    to generate oxygen from the Martian atmosphere.",
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
            CONTEXT: "The Mediterranean diet is a commonly recommended dietary plan that emphasizes fruits, \
                vegetables, whole grains, legumes, lean proteins, and healthy fats. Studies have shown that it \
                    offers numerous health benefits, including a reduced risk of heart disease and improved \
                        cognitive health.",
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
            CONTEXT: "The Queen's Royal Castle is a well-known tourist attraction in the United Kingdom. It spans \
                over 500 acres and contains extensive gardens and parks. The castle was built in the 15th century \
                    and has been home to generations of royalty.",
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

FLUENCY_ANNOTATION_TEMPLATE = "\n\n".join(
    [
        "System:",
        "You are an AI assistant. You will be given the definition of an evaluation metric for assessing the quality \
            of an answer in a question-answering task. Your job is to compute an accurate evaluation score using the \
                provided evaluation metric.",
        f"Fluency measures the quality of individual sentences in the answer, and whether they are well-written and \
            grammatically correct. Consider the quality of individual sentences when evaluating fluency. Given the \
                question and answer, score the fluency of the answer between {MIN_RATING} to {MAX_RATING} using the \
                    following {RATING} scale:",
        f"{RATING} 1: the answer completely lacks fluency",
        f"{RATING} 2: the answer mostly lacks fluency",
        f"{RATING} 3: the answer is partially fluent",
        f"{RATING} 4: the answer is mostly fluent",
        f"{RATING} 5: the answer has perfect fluency",
        f"The score should be integer only, between {MIN_RATING} and {MAX_RATING}.",
        "## Example Task #0",
        json.dumps({
            PROMPT: "What did you have for breakfast today?",
            COMPLETION: "Breakfast today, me eating cereal and orange juice very good.",
        }),
        "A good example response would be:",
        "## Example Task #0:",
        json.dumps({
            RATING: 1
        }),
        "## Example Task #1",
        json.dumps({
            PROMPT: "How do you feel when you travel alone?",
            COMPLETION: "Alone travel, nervous, but excited also. I feel adventure and like its time.",
        }),
        "A good example response would be:",
        "## Example Task #1:",
        json.dumps({
            RATING: 2
        }),
        "## Example Task #2",
        json.dumps({
            PROMPT: "When was the last time you went on a family vacation?",
            COMPLETION: "Last family vacation, it took place in last summer. We traveled to a beach destination, \
                very fun.",
        }),
        "A good example response would be:",
        "## Example Task #2:",
        json.dumps({
            RATING: 3
        }),
        "## Example Task #3",
        json.dumps({
            PROMPT: "What is your favorite thing about your job?",
            COMPLETION: "My favorite aspect of my job is the chance to interact with diverse people. I am constantly \
                learning from their experiences and stories.",
        }),
        "A good example response would be:",
        "## Example Task #3:",
        json.dumps({
            RATING: 4
        }),
        "## Example Task #4",
        json.dumps({
            PROMPT: "Can you describe your morning routine?",
            COMPLETION: "Every morning, I wake up at 6 am, drink a glass of water, and do some light stretching. \
                After that, I take a shower and get dressed for work. Then, I have a healthy breakfast, usually \
                    consisting of oatmeal and fruits, before leaving the house around 7:30 am.",
        }),
        "A good example response would be:",
        "## Example Task #4:",
        json.dumps({
            RATING: 5
        }),
        "User:",
        "{input_samples}"
    ]
)
COHERENCE_ANNOTATION_TEMPLATE = "\n\n".join(
    [
        "System:",
        "You are an AI assistant. You will be given the definition of an evaluation metric for assessing the quality \
            of an answer in a question-answering task. Your job is to compute an accurate evaluation score using \
                the provided evaluation metric.",
        f"Coherence of an answer is measured by how well all the sentences fit together and sound naturally as a \
            whole. Consider the overall quality of the answer when evaluating coherence. Given the question and \
                answer, score the fluency of the answer between {MIN_RATING} to {MAX_RATING} using the following \
                    {RATING} scale:",
        f"{RATING} 1: the answer completely lacks coherence",
        f"{RATING} 2: the answer mostly lacks coherence",
        f"{RATING} 3: the answer is partially coherent",
        f"{RATING} 4: the answer is mostly coherent",
        f"{RATING} 5: the answer has perfect coherence",
        f"The score should be integer only, between {MIN_RATING} and {MAX_RATING}.",
        "## Example Task #0",
        json.dumps({
            PROMPT: "What is your favorite indoor activity and why do you enjoy it?",
            COMPLETION: "I like pizza. The sun is shining."
        }),
        "A good example response would be:",
        "## Example Task #0:",
        json.dumps({
            RATING: 1
        }),
        "## Example Task #1",
        json.dumps({
            PROMPT: "Can you describe your favorite movie without giving away any spoilers?",
            COMPLETION: "It is a science fiction movie. There are dinosaurs. The actors eat cake. People must \
                stop the villain."
        }),
        "A good example response would be:",
        "## Example Task #1:",
        json.dumps({
            RATING: 2
        }),
        "## Example Task #2",
        json.dumps({
            PROMPT: "What are some benefits of regular exercise?",
            COMPLETION: "Regular exercise improves your mood. A good workout also helps you sleep better. \
                Trees are green",
        }),
        "A good example response would be:",
        "## Example Task #2:",
        json.dumps({
            RATING: 3
        }),
        "## Example Task #3",
        json.dumps({
            PROMPT: "How do you cope with stress in your daily life?",
            COMPLETION: "I usually go for a walk to clear my head. Listening to music helps me relax as well. \
                Stress is a part of life, but we can manage it through some activities.",
        }),
        "A good example response would be:",
        "## Example Task #3:",
        json.dumps({
            RATING: 4
        }),
        "## Example Task #4",
        json.dumps({
            PROMPT: "What can you tell me about climate change and its effects on the environment?",
            COMPLETION: "Climate change has far-reaching effects on the environment. Rising temperatures result \
                in the melting of polar ice caps, contributing to sea-level rise. Additionally, more frequent and \
                    severe weather events, such as hurricanes and heatwaves, can cause disruption to ecosystems \
                        and human societies alike.",
        }),
        "A good example response would be:",
        "## Example Task #4:",
        json.dumps({
            RATING: 5
        }),
        "User:",
        "{input_samples}"
    ])
SIMILARITY_ANNOTATION_TEMPLATE = "\n\n".join(
    [
        "System:",
        "You are an AI assistant. You will be given the definition of an evaluation metric for assessing the \
            quality of an answer in a question-answering task. Your job is to compute an accurate evaluation \
                score using the provided evaluation metric.",
        f"Equivalence, as a metric, measures the similarity between the predicted answer and the correct answer. \
            If the information and content in the predicted answer is similar or equivalent to the correct answer, \
                then the value of the Equivalence metric should be high, else it should be low. Given the question, \
                    correct answer, and predicted answer, determine the value of Equivalence metric using the \
                        following rating scale between {MIN_RATING} to {MAX_RATING} using the following {RATING} \
                            scale:",
        f"{RATING} 1: the predicted answer is not at all similar to the correct answer",
        f"{RATING} 2: the predicted answer is mostly not similar to the correct answer",
        f"{RATING} 3: the predicted answer is somewhat similar to the correct answer",
        f"{RATING} 4: the predicted answer is mostly similar to the correct answer",
        f"{RATING} 5: the predicted answer is completely similar to the correct answer",
        f"The score should be integer only, between {MIN_RATING} and {MAX_RATING}.",
        "## Example Task #0",
        json.dumps({
            PROMPT: "What is the role of ribosomes?",
            COMPLETION: "Ribosomes participate in carbohydrate breakdown by removing nutrients from complex sugar \
                molecules.",
            GROUND_TRUTH: "Ribosomes are cellular structures responsible for protein synthesis. They interpret the \
                genetic information carried by messenger RNA (mRNA) and use it to assemble amino acids into proteins."
        }),
        "A good example response would be:",
        "## Example Task #0:",
        json.dumps({
            RATING: 1
        }),
        "## Example Task #1",
        json.dumps({
            PROMPT: "Why did the Titanic sink?",
            COMPLETION: "The sinking of the Titanic was a result of a large iceberg collision. This caused the ship \
                to take on water and eventually sink, leading to the death of many passengers due to a shortage of \
                    lifeboats and insufficient rescue attempts.",
            GROUND_TRUTH: "The Titanic sank after it struck an iceberg during its maiden voyage in 1912. The impact \
                caused the ship\'s hull to breach, allowing water to flood into the vessel. The ship\'s design, \
                    lifeboat shortage, and lack of timely rescue efforts contributed to the tragic loss of life."
        }),
        "A good example response would be:",
        "## Example Task #1:",
        json.dumps({
            RATING: 2
        }),
        "## Example Task #2",
        json.dumps({
            PROMPT: "What causes seasons on Earth?",
            COMPLETION: "Seasons occur because of the Earth\'s rotation and its elliptical orbit around the Sun. \
                The tilt of the Earth\'s axis causes regions to be subjected to different sunlight intensities, \
                    which leads to temperature fluctuations and alternating weather conditions.",
            GROUND_TRUTH: "Seasons on Earth are caused by the tilt of the Earth\'s axis and its revolution around \
                the Sun. As the Earth orbits the Sun, the tilt causes different parts of the planet to receive \
                    varying amounts of sunlight, resulting in changes in temperature and weather patterns."
        }),
        "A good example response would be:",
        "## Example Task #2:",
        json.dumps({
            RATING: 3
        }),
        "## Example Task #3",
        json.dumps({
            PROMPT: "How does photosynthesis work?",
            COMPLETION: "In photosynthesis, sunlight is transformed into nutrients by plants and certain \
                microorganisms. Light is captured by chlorophyll molecules, followed by the conversion of \
                    carbon dioxide and water into sugar and oxygen through multiple reactions.",
            GROUND_TRUTH: "Photosynthesis is a process by which green plants and some other organisms convert \
                light energy into chemical energy. This occurs as light is absorbed by chlorophyll molecules, and \
                    then carbon dioxide and water are converted into glucose and oxygen through a series of reactions."
        }),
        "A good example response would be:",
        "## Example Task #3:",
        json.dumps({
            RATING: 4
        }),
        "## Example Task #4",
        json.dumps({
            PROMPT: "What are the health benefits of regular exercise?",
            COMPLETION: "Routine physical activity can contribute to maintaining ideal body weight, enhancing \
                muscle and bone strength, and preventing chronic illnesses. In addition, it supports mental \
                    health by alleviating stress and augmenting general mood.",
            GROUND_TRUTH: "Regular exercise can help maintain a healthy weight, increase muscle and bone strength, \
                and reduce the risk of chronic diseases. It also promotes mental well-being by reducing stress and \
                    improving overall mood."
        }),
        "A good example response would be:",
        "## Example Task #4:",
        json.dumps({
            RATING: 5
        }),
        "User:",
        "{input_samples}"
    ])
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

ANNOTATION_TEMPLATES = {
    GROUNDEDNESS: GROUNDING_ANNOTATION_TEMPLATE,
    RELEVANCE: RELEVANCE_ANNOTATION_TEMPLATE,
    FLUENCY: FLUENCY_ANNOTATION_TEMPLATE,
    COHERENCE: COHERENCE_ANNOTATION_TEMPLATE,
    SIMILARITY: SIMILARITY_ANNOTATION_TEMPLATE
}

OUTPUT_SPLITTING_REGEX = r"[# ]*Task #*\d+:?"

AUTHORIZATION = "Authorization"
BEARER = "Bearer"
API_KEY = "api-key"

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
        raise RuntimeError(f"Invalid Azure endpoint domain URL: {domain}.")

    url = url_pattern.format(domain, model)

    if api_version:
        url += f"?api-version={api_version}"

    return url


class _JobStatus(Enum):
    SUCCESS = "success"
    ENDPOINT_ERROR = "endpoint error"
    PARSING_FAILED = "parsing failed"


class _TokenScope(Enum):
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

                self.api_version = connection.metadata["ApiVersion"]
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

    def __init__(self):
        self.encoding = tiktoken.get_encoding(CL_100K_BASE)

    def count_tokens(self, input_str: str) -> int:
        # Count tokens, including special tokens like <|endofprompt|>
        return len(self.encoding.encode(input_str, allowed_special="all"))


class _PromptBuilder:
    def __init__(
        self,
        template: str,
        template_requirements: List[str],
        max_tokens: int,
        min_input_examples: int = 1,
    ):
        """Batches prompts according to limits."""
        self.template = template
        self.template_requirements = template_requirements
        self.max_tokens = max_tokens
        self.min_input_examples = min_input_examples
        self.tokenizer = _Tokenizer()

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
            input_examples = input_data[next_index: stop_index]

            # Build prompt given input, shot, and token limits
            prompt, n_tokens, n_inputs = self.build_prompt_with_limits(
                input_examples
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
                n_tokens_estimate=n_tokens
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
        """Build prompt from input_examples using template. Encode examples into JSON format.

        Args:
            input_examples: list of input examples

        Returns:
            prompt: constructed prompt
        """
        input_samples = [
            f"## Task #{i}:\n{self.encode_example(input_example, self.template_requirements)}"
            for (i, input_example) in enumerate(input_examples, start=1)
        ]
        t = self.template.replace("{input_samples}", "\n".join(input_samples))
        return t


def _split_output_examples(output_str: str) -> List[str]:
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


class _Job:
    """Job state for prompts submitted to model engines."""

    def __init__(
        self,
        job_idx: int,
        prompt_data: _PromptData,
        request_params: dict,
        retries_attempted: int = 0,
        response_data: dict = None,
        status: _JobStatus = None,
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
):
    # if we count too many errors, we stop and raise an exception
    error_count = 0
    request_count = 0
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
            "messages": [
                {"role": "user", "content": job.prompt_data.prompt}
            ],
        }

        if job.request_params.get("stop", None):
            request_data["stop"] = job.request_params["stop"]

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


def _request_prompt_batch(
    jobs: List[_Job],
    token_manager: _APITokenManager,
    model: str,
    azure_endpoint_domain_name: str,
    azure_openai_api_version: str,
    request_error_rate_threshold: float = 0.5,
    api_call_retry_backoff_factor: int = 3,
    api_call_retry_max_count: int = 3,
    **kwargs,
) -> List[_Job]:

    azure_endpoint_url = _check_and_format_azure_endpoint_url(
        AZURE_OPENAI_API_COMPLETION_URL_PATTERN,
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

        _query_inference_for_line(
            jobs,
            session,
            endpoint_url,
            token_manager,
            request_error_rate_threshold
        )


def _decode_example(
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
            raise ValueError(
                f"Label key {label_key} not in output example"
            )

    return example


def _parse_responses(
    job: _Job,
    num_samples: int,
):
    """Given a stream of job responses, parse them."""
    job.response_data["label_keys"] = LABEL_KEYS

    n_inputs = len(job.prompt_data.input_idx)

    if job.response_data["finish_reason"] == ["error"]:
        job.status = _JobStatus.ENDPOINT_ERROR
        print(
            "Cannot parse job due to finish_reason=error: "
            f"{job.response_data}")
        return

    assert len(job.response_data["samples"]) == num_samples, (
        f"Expected {num_samples} samples, "
        f"got {len(job.response_data['samples'])}"
    )

    output_examples = []
    num_failed = 0
    for sample in job.response_data["samples"]:
        # try to split the output into {num_samples} examples
        try:
            sample_examples = (
                _split_output_examples(sample)
            )

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
                    _decode_example(example, LABEL_KEYS)
                )
            output_examples.append(sample_examples_parsed)
        except Exception as _:  # noqa: B902, F841
            output_examples.append(None)
            print(f"Failed decoding examples: {sample_examples}")
            num_failed += 1

    # if the number of expected samples does not match the output samples
    # there was an issue getting the response and we cannot accurately map
    # the indices with the input data
    index_mapping = job.prompt_data.input_idx
    if output_examples[0] is None:
        print("Not all responses could be parsed correctly. Ignoring indices for violations")
        index_mapping = [-1] * len(index_mapping)

    if num_failed == num_samples:
        job.status = _JobStatus.PARSING_FAILED
    else:
        job.status = _JobStatus.SUCCESS
        job.response_data["output_examples"] = output_examples
        job.response_data["index_mapping"] = index_mapping


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
        token_manager: _APITokenManager,
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

        _request_prompt_batch(
            jobs=jobs, token_manager=token_manager, **endpoint_args
        )

        responses_to_print = "\n".join(
            [f"    Response: {job.response_data}" for job in jobs]
        )
        print(f"Received responses from annotation: \n{responses_to_print}\n")

        # Parse responses for this batch of jobs
        for job in jobs:
            _parse_responses(job, num_samples)

        # TODO handle errors and attempt retries
        job_results = [
            result
            for job in jobs
            if job.status == _JobStatus.SUCCESS
            for result in job.response_data["output_examples"][0]
        ]
        job_indices = [
                result
                for job in jobs
                if job.status == _JobStatus.SUCCESS
                for result in job.response_data["index_mapping"]
        ]
        return pd.DataFrame({
            RATING: [res[RATING] for res in job_results],
            INDEX: job_indices,
        })

    def make_job(self, **job_params) -> _Job:
        """Make job, tracking job_idx internally."""
        new_job = _Job(job_idx=self.job_idx, **job_params)
        self.job_idx += 1
        return new_job


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
    parser.add_argument("--api_call_retry_backoff_factor", type=int, default=3)
    parser.add_argument("--api_call_retry_max_count", type=int, default=10)
    parser.add_argument("--histogram", type=str, required=True)
    parser.add_argument("--samples_index", type=str, required=True)
    parser.add_argument("--groundedness_violations", type=str, required=True)
    parser.add_argument("--fluency_violations", type=str, required=True)
    parser.add_argument("--relevance_violations", type=str, required=True)
    parser.add_argument("--coherence_violations", type=str, required=True)
    parser.add_argument("--similarity_violations", type=str, required=True)

    parser.add_argument("--workspace_connection_arm_id", type=str, required=True)
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

    input_metric_names = [m.strip() for m in args.metric_names.split(",")]

    if not (set(input_metric_names) <= set(ALL_METRIC_NAMES)):
        raise ValueError(
            f"metric_names must be a comma-separated list of metric names "
            f"and a subset of {ALL_METRIC_NAMES}, got {args.metric_names}."
        )

    # remove all but groundedness/fluency/coherence/relevance/similarity from metric names and
    # remove duplicates
    pruned_metric_names = [re.sub(r'^(.*?)(Groundedness|Fluency|Coherence|Relevance|Similarity)(.*?)$', r'\2', m) for
                           m in input_metric_names]
    metric_names = list(set(pruned_metric_names))

    # Validate inputs
    if args.temperature < 0.0 or args.temperature > 2.0:
        raise ValueError(f"temperature must be between 0.0 and 2.0, inclusive; "
                         f"got {args.temperature}.")
    if args.top_p < 0.0 or args.top_p > 1.0:
        raise ValueError(f"top_p must be between 0.0 and 1.0, inclusive; got {args.top_p}.")
    if args.num_samples <= 0:
        # TODO support multiple returned annotations
        raise ValueError(f"num_samples must be 1, got {args.num_samples}.")
    if args.frequency_penalty < -2.0 or args.frequency_penalty > 2.0:
        raise ValueError(
            "frequency_penalty must be between -2.0 and 2.0, inclusive; "
            f"got {args.frequency_penalty}."
        )
    if args.presence_penalty < -2.0 or args.presence_penalty > 2.0:
        raise ValueError(
            f"presence_penalty must be between -2.0 and 2.0, inclusive; "
            f"got {args.presence_penalty}."
        )

    if args.sample_rate <= 0.0 or args.sample_rate > 1.0:
        raise ValueError(f"sample_rate must be larger than 0.0 and at most 1.0, "
                         f"got {args.sample_rate}.")

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
        metric_names=metric_names,
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
    )


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
):
    """Apply annotation to all samples in the production_dataset."""
    production_df = io_utils.read_mltable_in_spark(production_dataset)
    # Ensure input data has the correct columns given the metrics
    # Question, answer required for coherence and fluency
    qa_required = len(list(set(QA_METRIC_NAMES).intersection(
        set(metric_names))))
    for col_name in [prompt_column_name, completion_column_name]:
        if col_name not in production_df.columns and qa_required:
            raise ValueError(f"production_dataset must have column: {col_name}")
    # Question, answer, context required for relevance and groundedness
    qac_required = len(list(set(QAC_METRIC_NAMES).intersection(
        set(metric_names))))
    if qac_required and context_column_name not in production_df.columns:
        raise ValueError(f"production_dataset must have column: {context_column_name}")
    # Question, answer, ground-truth required for similarity
    if SIMILARITY in metric_names and ground_truth_column_name not in production_df.columns:
        raise ValueError(f"production_dataset must have column: {ground_truth_column_name}")

    annotation_requirements = {
        GROUNDEDNESS: [prompt_column_name, completion_column_name, context_column_name],
        RELEVANCE: [prompt_column_name, completion_column_name, context_column_name],
        FLUENCY: [prompt_column_name, completion_column_name],
        COHERENCE: [prompt_column_name, completion_column_name],
        SIMILARITY: [prompt_column_name, completion_column_name, ground_truth_column_name]
    }
    # Sampling
    production_df_sampled = production_df.sample(withReplacement=False, fraction=sample_rate)
    if production_df.count() == 0:
        print("Not enough data resulting from sample_rate and production dataset. "
                         "Using default of 5 records. To use sample_rate with this dataset, "
                         "try increasing sample_rate value.")
        production_df_sampled.sample(withReplacement=False, n=5)
    production_df = production_df_sampled
    row_count = production_df.count()
    production_df_with_index = production_df_sampled.withColumn("id",
                                                        row_number()
                                                        .over(Window.orderBy(monotonically_increasing_id()))-1)

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

    print(
        "Created token manager for auth type "
        f"managed identity using auth header {API_KEY}."
    )
    endpoint_args["azure_endpoint_domain_name"] = endpoint_domain_name
    endpoint_args["azure_openai_api_version"] = api_version

    # use fixed API version since newer versions aren't supported
    get_model_endpoint = _check_and_format_azure_endpoint_url(AZURE_OPENAI_API_DEPLOYMENT_URL_PATTERN,
                                                              AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE,
                                                              endpoint_domain_name, "2022-12-01",
                                                              model_deployment_name)
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

    request_args["max_tokens"] = BASE_MAX_TOKENS * MODEL_TYPE_FACTOR[model_type]
    max_inputs = BASE_MAX_INPUTS * MODEL_TYPE_FACTOR[model_type]
    max_prompt_tokens = BASE_MAX_PROMPT_TOKENS * MODEL_TYPE_FACTOR[model_type]

    print(f"Using max_inputs = {max_inputs}")
    print(f"Using max_prompt_tokens = {max_prompt_tokens}")

    print(f"starting submit_inputs with endpoint_args={endpoint_args}")
    all_metrics_pdf = None
    samples_index_rows = []
    for metric_name in metric_names:
        # Run inference over input dataset
        print(f"Begin {metric_name} request.")

        def annotate_batch(iterator):
            # _PromptBuilder can iteratively batch unlabeled examples
            prompt_builder = _PromptBuilder(
                template=ANNOTATION_TEMPLATES[metric_name],
                template_requirements=annotation_requirements[metric_name],
                max_tokens=max_prompt_tokens,
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
                    token_manager=token_manager,
                    input_data_df=batch,
                    request_args=request_args,
                    max_inputs=max_inputs,
                    num_samples=num_samples,
                    **endpoint_args,
                )

        annotations_df = production_df.mapInPandas(
            annotate_batch,
            schema=StructType(
                [
                    StructField(RATING, IntegerType(), True),
                    StructField(INDEX, IntegerType(), True),
                ]
            ),
        ).cache()

        # Get rating counts
        metrics_df = annotations_df.select(RATING).groupBy(RATING).count()
        metrics_df.show()
        print("Finished annotating answers.")

        metric_name_compact = metric_name.replace(" ", "").title()

        metrics_pdf = metrics_df.select("*").toPandas()
        print(metrics_pdf)
        ratings = metrics_pdf.rating.to_list()
        missing_ratings = set(range(MIN_RATING, MAX_RATING + 1)) - set(ratings)
        for r in missing_ratings:
            metrics_pdf = metrics_pdf.append({RATING: r, COUNT: 0}, ignore_index=True)
        metrics_pdf[RATING] = metrics_pdf[RATING].map(lambda r: str(r))
        # add metric_name, metric_value, group, and threshold values
        metrics_pdf.rename(columns={RATING: GROUP, COUNT: METRIC_VALUE, }, inplace=True)
        metrics_pdf[METRIC_NAME] = f"Acceptable{metric_name_compact}ScorePerInstance"
        metric_threshold_value = str(threshold_args[f"{metric_name_compact.lower()}_rating_threshold"])
        metrics_pdf[THRESHOLD] = metric_threshold_value
        print(metrics_pdf)

        # create violations table if there are violations
        violations_df = annotations_df.filter((col(RATING) < metric_threshold_value) & (col(INDEX) != -1))
        if violations_df.count() > 0:
            violations_df_full = production_df_with_index.join(violations_df,
                                                               production_df_with_index.id == violations_df.index,
                                                               "inner").drop(violations_df.index).drop(
                                                                   production_df_with_index.id).withColumnRenamed(
                                                                       'rating', metric_name_compact)
            run_id = os.environ.get("AZUREML_RUN_ID")
            io_utils.save_spark_df_as_mltable(violations_df_full, violations[metric_name_compact.lower()])
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
        all_metrics_pdf = all_metrics_pdf.append(
            {
                METRIC_NAME: "RowCount",
                METRIC_VALUE: row_count,
                GROUP: row_count_name,
                THRESHOLD: ""
            },
            ignore_index=True)
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
    # Create a new DataFrame for the samples index data
    samples_df = spark.createDataFrame(samples_index_rows, metadata_schema)
    io_utils.save_spark_df_as_mltable(samples_df, samples_index)

    io_utils.save_spark_df_as_mltable(
        spark.createDataFrame(all_metrics_pdf),
        histogram)


if __name__ == "__main__":
    run()
