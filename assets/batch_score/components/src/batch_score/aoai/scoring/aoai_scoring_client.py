# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copied from https://github.com/Azure/azureai-insiders/tree/main/previews/batch-inference-using-aoai
# and then modified.

"""Scoring client targeting to Azure OpenAI endpoints."""

import asyncio
import json
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone

import aiohttp

from ...common.auth.auth_provider import (
    ApiKeyAuthProvider,
    AuthProvider,
    IdentityAuthProvider,
    WorkspaceConnectionAuthProvider,
)
from ...common.configuration.configuration import Configuration
from ...common.scoring.scoring_request import ScoringRequest
from ...common.scoring.scoring_result import (
    RetriableException,
    ScoringResult,
    ScoringResultStatus,
)
from ...common.scoring.tally_failed_request_handler import TallyFailedRequestHandler
from ...common.telemetry.logging_utils import get_logger
from ...common.telemetry.scoring_logging import (
    ScoreFailedLog,
    ScoreFailedWithExceptionLog,
    ScoreStartLog,
    ScoreSucceedLog,
)


class UnrecognizedScoringApiException(Exception):
    """Unrecognized scoring api exception."""

    def __init__(self, scoring_url: str, expected_scoring_apis: 'list[str]'):
        """Init function."""
        self.scoring_url = scoring_url
        self.expected_scoring_apis = expected_scoring_apis

    def __str__(self):
        """Str function."""
        return f"Unrecognized scoring URL '{self.scoring_url}' does not contain any of {self.expected_scoring_apis}."


# TO DO: Rename to LLMScoringClient
class AoaiScoringClient:
    """AOAI scoring client."""

    RETRIABLE_STATUS_CODE = [408, 429, 500, 502, 503, 504]

    EMBEDDING_API = "/embeddings"
    CHAT_COMPLETION_API = "/chat/completions"
    COMPLETION_API = "/completions"

    # reference: https://platform.openai.com/docs/api-reference
    SAMPLE_PAYLOADS = {
        EMBEDDING_API: {
            "input": "The food was delicious and the waiter..."
        },
        CHAT_COMPLETION_API: {
            "max_tokens": 5,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Hello!"
                }
            ]
        },
        COMPLETION_API: {
            "max_tokens": 5,
            "prompt": "Hello, my name is"
        },
    }

    def __init__(
            self,
            auth_provider: AuthProvider,
            scoring_url: str = None,
            tally_handler: TallyFailedRequestHandler = None,
            additional_headers: str = None):
        """Init function."""
        self.__auth_provider: AuthProvider = auth_provider
        self.__scoring_url: str = scoring_url
        self.__tally_handler = tally_handler

        if additional_headers is not None:
            self.__additional_headers = json.loads(additional_headers)
        else:
            self.__additional_headers = {}

    async def score_once(self,
                         session: aiohttp.ClientSession,
                         scoring_request: ScoringRequest,
                         # TODO: use timeout and worker id
                         timeout: aiohttp.ClientTimeout = None,
                         worker_id: str = "1") -> ScoringResult:
        """Score a single request until terminal status is reached."""
        response = None
        response_payload = None
        response_status = None
        response_headers = None
        model_response_code = None
        error = None
        is_retriable = False

        start = time.time()

        headers = self.__get_headers()
        get_logger().debug(headers)

        ScoreStartLog(internal_id=scoring_request.internal_id,
                      x_ms_client_request_id=headers["x-ms-client-request-id"],
                      timestamp=datetime.fromtimestamp(start, timezone.utc),
                      scoring_url=self.__scoring_url).log()

        try:
            # request_body = json.dumps(scoring_request.request_payload)
            request_body = scoring_request.cleaned_payload
            async with session.post(url=self.__scoring_url, headers=headers, data=request_body) as response:
                response_status = response.status
                response_headers = response.headers
                model_response_code = response_headers.get("ms-azureml-model-error-statuscode")

                if response_status == 200:
                    response_payload = await response.json()

                else:
                    is_retriable = response_status in AoaiScoringClient.RETRIABLE_STATUS_CODE
                    error_response = await response.text()

                    try:
                        response_payload = json.loads(error_response)
                    except Exception:
                        response_payload = error_response

                    error = f"request failed with status {response_status}."

                    ScoreFailedLog(internal_id=scoring_request.internal_id,
                                   x_ms_client_request_id=headers["x-ms-client-request-id"],
                                   status_code="NONE" if response is None else response.status,
                                   reason="NONE" if response is None else response.reason,
                                   response_headers="NONE" if response is None else response.headers,
                                   response_payload=error_response).log()

        except (aiohttp.ServerConnectionError, aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            is_retriable = True
            error = traceback.format_exc()

            ScoreFailedWithExceptionLog(internal_id=scoring_request.internal_id,
                                        x_ms_client_request_id=headers["x-ms-client-request-id"],
                                        exception_type=type(e).__name__,
                                        exception=e).log()

        except Exception as e:
            error = traceback.format_exc()

            ScoreFailedWithExceptionLog(internal_id=scoring_request.internal_id,
                                        x_ms_client_request_id=headers["x-ms-client-request-id"],
                                        exception_type=type(e).__name__,
                                        exception=error,
                                        unhandled_exc=True).log()

        end = time.time()

        result: ScoringResult = None

        if response_status == 200:
            ScoreSucceedLog(internal_id=scoring_request.internal_id,
                            x_ms_client_request_id=headers["x-ms-client-request-id"],
                            duration=end-start).log()

            result = ScoringResult(
                status=ScoringResultStatus.SUCCESS,
                start=start,
                end=end,
                request_obj=scoring_request.original_payload_obj,
                request_metadata=scoring_request.request_metadata,
                response_body=response_payload,
                response_headers=response_headers,
                num_retries=0,
                token_counts=scoring_request.estimated_token_counts
            )

        elif is_retriable:
            raise RetriableException(status_code=response_status, response_payload=response_payload)

        else:  # Score failed
            result = ScoringResult(
                status=ScoringResultStatus.FAILURE,
                start=start,
                end=end,
                request_obj=scoring_request.original_payload_obj,
                request_metadata=scoring_request.request_metadata,
                response_body=response_payload,
                response_headers=response_headers,
                num_retries=0
            )

        if result.status == ScoringResultStatus.FAILURE and \
                self.__tally_handler.should_tally(response_status=response_status,
                                                  model_response_status=model_response_code):
            result.omit = True

        return result

    def __get_headers(self) -> "dict[str, any]":
        """Retrieve request headers."""
        headers = {
            'Content-Type': 'application/json',
            'x-ms-client-request-id': str(uuid.uuid4())
        }

        auth_headers = self.__auth_provider.get_auth_headers()

        headers.update(auth_headers)
        headers.update(self.__additional_headers)

        return headers

    @staticmethod
    def create(
        configuration: Configuration,
        tally_handler: TallyFailedRequestHandler
    ) -> 'AoaiScoringClient':
        """Create a client using command line arguments."""
        auth_provider: AuthProvider = None

        if configuration.authentication_type == "api_key":
            auth_provider = ApiKeyAuthProvider(configuration.api_key_name)
        elif configuration.authentication_type == "managed_identity":
            auth_provider = IdentityAuthProvider(use_user_identity=False)
        elif configuration.authentication_type in ["azureml_workspace_connection", "connection"]:
            endpoint_type = configuration.get_endpoint_type()
            auth_provider = WorkspaceConnectionAuthProvider(configuration.connection_name, endpoint_type)
        else:
            raise Exception(f"Invalid authentication type {configuration.authentication_type}")

        scoring_client = AoaiScoringClient(
            auth_provider=auth_provider,
            scoring_url=configuration.scoring_url,
            tally_handler=tally_handler,
            additional_headers=None
        )

        return scoring_client

    def validate(self):
        """Validate the scoring client by scoring a sample."""
        try:
            get_logger().info("Validating the scoring client by scoring a sample.")
            result = asyncio.run(self._score_sample())
        except UnrecognizedScoringApiException as e:
            get_logger().warning("Could not validate the scoring client. Exception: " + str(e))
            return

        if result.status == ScoringResultStatus.SUCCESS:
            get_logger().debug("Successfully scored a sample.")
            return

        error_msg = f"Failed to score a sample. Error: {json.dumps(result.response_body)}"
        get_logger().error(error_msg)
        raise Exception(error_msg)

    async def _score_sample(self):
        """Score with a sample request."""
        api = AoaiScoringClient._infer_api(self.__scoring_url)
        scoring_request = AoaiScoringClient._get_sample_scoring_request(api)

        timeout_duration = timedelta(seconds=10).total_seconds()
        timeout = aiohttp.ClientTimeout(total=timeout_duration)

        async with aiohttp.ClientSession() as session:
            return await self.score_once(
                session,
                scoring_request,
                timeout,
                worker_id="confirm_scoring_endpoint_accessibility")

    @staticmethod
    def _infer_api(scoring_url: str) -> ScoringRequest:
        """Infer the scoring API type from the scoring URL."""
        # Order matters. The first one that matches is returned.
        # Chat completion API must be checked before completion API.
        expected_scoring_apis = [
            AoaiScoringClient.EMBEDDING_API,
            AoaiScoringClient.CHAT_COMPLETION_API,
            AoaiScoringClient.COMPLETION_API,
        ]

        for api in expected_scoring_apis:
            if api in scoring_url:
                return api

        raise UnrecognizedScoringApiException(scoring_url, expected_scoring_apis)

    @staticmethod
    def _get_sample_scoring_request(api: str) -> ScoringRequest:
        """Get a sample scoring request."""
        payload = json.dumps(AoaiScoringClient.SAMPLE_PAYLOADS[api])
        return ScoringRequest(original_payload=payload)
