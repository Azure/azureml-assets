import asyncio
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict

from aiohttp import (
    ClientConnectionError,
    ClientResponseError,
    ClientSession,
    ContentTypeError,
    ServerConnectionError,
)

from ...common.scoring.scoring_request import ScoringRequest
from ...common.scoring.scoring_result import (
    PermanentException,
    RetriableException,
    ScoringResult,
    ScoringResultStatus,
)
from ...common.telemetry import logging_utils as lu
from ...common.telemetry.logging_utils import get_events_client
from ...header_handlers.rate_limiter.rate_limiter_header_handler import (
    RateLimiterHeaderHandler,
)
from ...utils import common
from .estimators import *


class QuotaClient:
    ESTIMATORS = {
        "completion": CompletionEstimator,
        "chat_completion": ChatCompletionEstimator,
        "embeddings": EmbeddingsEstimator,
        "vesta": VestaEstimator,
    }

    FATAL_RESPONSE_CODES = [
        # Returned if:
        #  - The quota audience doesn't exist.
        #  - Our client is buggy and sent a bad request.
        400,

        # Returned for auth errors.
        401, 403,

        # Returned in some cases for bad audiences.
        404,
    ]

    MAX_ATTEMPTS = 5

    RETRIABLE_RESPONSE_CODES = [500]

    def __init__(self,
                 header_handler: RateLimiterHeaderHandler,
                 service_namespace: str,
                 quota_audience: str,
                 batch_pool: str,
                 quota_estimator: str):
        self.__header_handler = header_handler
        self.__namespace = service_namespace
        self.__audience = quota_audience
        self.__batch_pool = batch_pool
        self.__estimator = self.ESTIMATORS[quota_estimator]()

        self.__base_url = os.environ.get("BATCH_SCORE_QUOTA_BASE_URL",
                                         "https://azureml-drl-us.azureml.ms/ratelimiter")

        self.__enabled = bool(self.__namespace and self.__audience and self.__batch_pool)
        if not self.__enabled and (self.__namespace or self.__audience or self.__batch_pool):
            raise Exception("Batch pools and quota enforcement can only be used together. "
                            "You must also provide both a service namespace and audience for quota enforcement."
                            f"Namespace: {self.__namespace}; "
                            f"Audience: {self.__audience}; "
                            f"Batch pool: {self.__batch_pool}")

    @asynccontextmanager
    async def reserve_capacity(self, session: ClientSession, scope: str, request: ScoringRequest):
        """Request capacity from the quota service and release it when finished."""

        if request.estimated_cost == 0:
            estimated_cost = self.__estimator.estimate_request_cost(request.cleaned_payload_obj)
            if isinstance(estimated_cost, int):
                request.estimated_cost = estimated_cost
            else:
                # Embeddings: Record the token count of each input in the batch, then
                # set the estimated cost of the entire batched request to 1.
                request.estimated_token_counts = estimated_cost
                request.estimated_cost = 1

        lease = await self.__acquire_lease(session, scope, request.estimated_cost, request.internal_id, request.retry_count)

        try:
            yield lease
        finally:
            await lease.end()

    # read-only
    @property
    def batch_pool(self):
        return self.__batch_pool

    # read-only
    @property
    def audience(self):
        return self.__audience

    async def __acquire_lease(self, session: ClientSession, scope: str, capacity: int, request_id: str, retry_count: int):
        if not self.__enabled:
            return NullQuotaLease()

        for attempt in range(self.MAX_ATTEMPTS):
            try:
                response = await self._api_request_lease(session, scope, capacity, request_id, retry_count)
                return QuotaLease(self, session, scope, response, capacity, request_id)
            except Exception as e:
                if self.__exception_is_backpressure(e):
                    lu.get_logger().debug(f"QuotaClient: Throttled by quota service ({capacity} tokens requested).")
                    raise QuotaUnavailableException(retry_after=self.__exception_get_retry_after(e))
                elif self.__exception_is_fatal(e):
                    lu.get_logger().exception("QuotaClient: Fatal error response from quota service.")
                    raise QuotaPermanentlyUnavailableException(
                        scope,
                        self.__audience,
                        capacity,
                        self.__exception_get_status_code(e),
                        self.__exception_get_message(e))
                elif self.__exception_is_retriable(e):
                    lu.get_logger().debug(f"QuotaClient: Transient error while acquiring quota lease: {e}")
                    await asyncio.sleep(common.backoff(attempt))
                else:
                    # Something happened that we don't know how to handle.
                    # Rather than terminating the job or looping endlessly,
                    # we'll let the request go through and hope for the best.
                    lu.get_logger().exception("QuotaClient: Unexpected error while acquiring quota lease. Proceeding without one.")
                    return NullQuotaLease()

        lu.get_logger().error("QuotaClient: Retries exhausted while acquiring quota lease. Proceeding without one.")
        return NullQuotaLease()

    def __exception_get_message(self, exception):
        if isinstance(exception, ClientResponseError):
            return exception.message
        else:
            return None

    def __exception_get_retry_after(self, exception):
        if isinstance(exception, ClientResponseError):
            if (x_ms_retry_after_ms := exception.headers.get("x-ms-retry-after-ms")):
                try:
                    # Unit is milliseconds, normalize to seconds.
                    return float(x_ms_retry_after_ms) / 1000
                except:
                    lu.get_logger().debug(f"QuotaClient: Cannot parse x-ms-retry-after-ms: {x_ms_retry_after_ms}")
            elif (retry_after := exception.headers.get("Retry-After")):
                try:
                    return float(retry_after)
                except:
                    lu.get_logger().debug(f"QuotaClient: Cannot parse Retry-After: {retry_after}")

        return None

    def __exception_get_status_code(self, exception):
        if isinstance(exception, ClientResponseError):
            return exception.status
        else:
            return None

    def __exception_is_backpressure(self, exception):
        if isinstance(exception, ClientResponseError):
            return exception.status == 429
        else:
            return False

    def __exception_is_fatal(self, exception):
        if isinstance(exception, ClientResponseError):
            return exception.status in self.FATAL_RESPONSE_CODES
        else:
            return False

    def __exception_is_retriable(self, exception):
        if isinstance(exception, ClientConnectionError):
            return True
        elif isinstance(exception, ClientResponseError):
            return exception.status in self.RETRIABLE_RESPONSE_CODES
        elif isinstance(exception, ServerConnectionError):
            return True
        else:
            return False

    def __get_headers(self, *, request_id: str = None, retry_count: int = None):
        additional_headers = {}

        if request_id is not None:
            additional_headers["x-ms-client-request-id"] = request_id

        if retry_count is not None:
            additional_headers["x-ms-retry-count"] = str(retry_count)

        return self.__header_handler.get_headers(additional_headers=additional_headers)

    def __get_url(self, scope: str, api_name: str) -> str:
        return f"{self.__base_url}/v1.0/servicenamespaces/{self.__namespace}/scopes/{scope}/audiences/{self.__audience}/{api_name}"

    async def _api_release_lease(self, session: ClientSession, scope: str, lease_id: str, request_id: str):
        url = self.__get_url(scope, "releaseLease")
        headers = self.__get_headers(request_id=request_id)

        body = {
            "leaseId": lease_id,
            "scope": scope,
        }

        async with session.post(url, headers=headers, json=body) as response:
            get_events_client().emit_quota_operation("ReleaseLease", response.status, lease_id=lease_id, amount=None, scoring_request_internal_id=request_id)
            response.raise_for_status()

    async def _api_renew_lease(self, session: ClientSession, scope: str, lease_id: str, request_id: str):
        url = self.__get_url(scope, "renewLease")
        headers = self.__get_headers(request_id=request_id)

        body = {
            "leaseId": lease_id,
            "scope": scope,
        }

        async with session.post(url, headers=headers, json=body) as response:
            get_events_client().emit_quota_operation("RenewLease", response.status, lease_id=lease_id, amount=None, scoring_request_internal_id=request_id)
            response.raise_for_status()
            try:
                return await response.json()
            except ContentTypeError:
                return None

    async def _api_request_lease(self, session: ClientSession, scope: str, amount: int, request_id: str, retry_count: int):
        url = self.__get_url(scope, "requestLease")
        headers = self.__get_headers(request_id=request_id, retry_count=retry_count)

        body = {
            "requestedCapacity": amount,
            "scope": scope,
        }

        async with session.post(url, headers=headers, json=body) as response:
            lease_id = None

            try:
                response.raise_for_status()
                lease = await response.json()
                lease_id = lease.get("leaseId") if isinstance(lease, dict) else None
                return lease
            finally:
                get_events_client().emit_quota_operation("RequestLease", response.status, lease_id=lease_id, amount=amount, scoring_request_internal_id=request_id)

    def _report_result(self, lease_id: int, quota_capacity: int, result: ScoringResult):
        try:
            if result.status == ScoringResultStatus.SUCCESS:
                result_cost = self.__estimator.estimate_response_cost(result.request_obj, result.response_body)
                #TODO: Eventify this trace
                # lu.get_logger().info(f"QuotaClient: Actual usage for quota lease {lease_id} was {result_cost} tokens "
                #                      f"({quota_capacity} tokens were reserved).")
        except Exception:
            lu.get_logger().exception("QuotaClient: Failed to report quota for scoring result.")


class QuotaLease:
    """Represents an active lease of token capacity from the quota service."""

    # Percentage of lease duration after which we'll automatically renew.
    RENEWAL_FRACTION = 0.7

    def __init__(self, client: QuotaClient, session: ClientSession, scope: str, lease: Dict[str, any], capacity: int, request_id: str):
        self.__client = client
        self.__session = session
        self.__scope = scope
        self.__capacity = capacity
        self.__request_id = request_id

        self.__set_lease(lease)

        lu.get_logger().debug(f"QuotaClient: Acquired quota lease {self.__lease_id} ({self.__capacity} tokens) until {self.__expiration}.")

        self.__task = asyncio.create_task(self.__keep_alive())

    async def end(self):
        """Terminates the background renewal task and releases the quota lease."""

        self.__task.cancel()

        try:
            await self.__client._api_release_lease(self.__session, self.__scope, self.__lease_id, self.__request_id)
            lu.get_logger().debug(f"QuotaClient: Released quota lease {self.__lease_id}.")
        except Exception:
            lu.get_logger().exception(f"QuotaClient: Failed to release quota lease {self.__lease_id}.")

    def report_result(self, result: ScoringResult):
        self.__client._report_result(self.__lease_id, self.__capacity, result)

    async def __keep_alive(self):
        """Background task that periodically renews the quota lease."""

        attempt = 0

        while True:
            try:
                now = datetime.now(timezone.utc)

                if now > self.__expiration:
                    lu.get_logger().error(f"QuotaClient: Quota lease {self.__lease_id} expired.")
                    break

                renewal = self.__expiration - (1 - self.RENEWAL_FRACTION) * self.__duration

                delay = common.backoff(attempt) # Try to wait this long ...
                delay = max(delay, (renewal - now).total_seconds()) # But no sooner than the renewal target ...
                delay = min(delay, (self.__expiration - now).total_seconds() - 1) # And no later than the expiration.

                await asyncio.sleep(delay)

                old_lease_id = self.__lease_id
                response = await self.__client._api_renew_lease(self.__session, self.__scope, self.__lease_id, self.__request_id)

                if response:
                    self.__set_lease(response)
                else:
                    self.__expiration = datetime.now(timezone.utc) + self.__duration

                attempt = 0

                lu.get_logger().debug(f"QuotaClient: Renewed quota lease {old_lease_id}; new lease {self.__lease_id} valid until {self.__expiration}.")
            except Exception:
                attempt += 1

                lu.get_logger().exception(f"QuotaClient: Failed to renew quota lease {self.__lease_id}.")

    def __parse_duration(self, value: str) -> timedelta:
        if (match := re.match(r"^(\d+):(\d+):(\d+)$", value)): # .NET TimeSpan default format
            return timedelta(hours=int(match.group(1)),
                             minutes=int(match.group(2)),
                             seconds=int(match.group(3)))
        else:
            raise Exception(f"Cannot parse quota lease duration: {value}")

    def __set_lease(self, lease: Dict[str, any]):
        self.__lease_id: str = lease["leaseId"]
        self.__duration: timedelta = self.__parse_duration(lease["leaseDuration"])

        # We expect the lease duration to be on the order of a few minutes,
        # and server/client clock skew could definitely exceed that. So we'll
        # set the expiration from local time and the duration, instead.
        self.__expiration: datetime = datetime.now(timezone.utc) + self.__duration


# Used as placeholder object when quota management is disabled.
class NullQuotaLease:
    async def end(self, *args, **kwargs):
        pass

    def report_result(self, *args, **kwargs):
        pass


class QuotaUnavailableException(RetriableException):
    def __init__(self, *, retry_after: float = None):
        super().__init__(429, retry_after=retry_after)


class QuotaPermanentlyUnavailableException(PermanentException):
    def __init__(self, scope: str, audience: str, capacity: int, status_code: int, message: str):
        message = f"Audience {audience} under scope {scope} does not have enough quota to satisfy a request of {capacity} tokens. Rate limiter service returned {status_code}: {message}"
        super().__init__(message, status_code)

        self.scope = scope
        self.audience = audience
        self.capacity = capacity
