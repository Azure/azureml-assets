# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The class for events client."""

import logging
import json
from contextvars import ContextVar
from opencensus.ext.azure.log_exporter import AzureEventHandler


class EventsClient:
    """Events client base class."""

    def emit_request_completed(
        self,
        latency: int,
        request_internal_id: str,
        client_request_id: str,
        endpoint_uri: str = "",
        status_code: str = "0",
        model_response_code: str = "",
        client_exception: str = "",
        is_retriable: bool = False
    ):
        """Interface for emit request."""
        pass

    def emit_tokens_generated(
        self,
        generated_tokens: int,
        context_tokens: int,
        endpoint_uri: str = "",
    ):
        """Interface for emit tokens generated."""
        pass

    def emit_request_concurrency(
        self,
        endpoint_uri: str,
        active_requests: int,
        estimated_cost: int
    ):
        """Interface for emit request concurrency."""
        pass

    def emit_worker_concurrency(
        self,
        worker_concurrency: int
    ):
        """Interface for emit worker concurrency."""
        pass

    def emit_row_completed(
        self,
        row_count: int,
        result: str = "SUCCESS"
    ):
        """Interface for emit row completed."""
        pass

    def emit_quota_operation(
        self,
        operation: str,
        status_code: int,
        lease_id: str,
        amount: int
    ):
        """Interface for emit quota operation."""
        pass

    def emit_mini_batch_started(
        self,
        input_row_count: int
    ):
        """Interface for emit mini batch started."""
        pass

    def emit_mini_batch_completed(
        self,
        input_row_count: int,
        output_row_count: int,
        exception: str = None,
        stacktrace: str = None
    ):
        """Interface for emit mini batch completed."""
        pass

    def emit_batch_driver_init(
        self,
        job_params: dict,
    ):
        """Interface for emit batch driver init."""
        pass

    def emit_batch_driver_shutdown(
        self,
        job_params: dict,
    ):
        """Interface for emit batch driver shutdown."""
        pass


class AppInsightsEventsClient(EventsClient):
    """Application insight events client class."""

    def __init__(self,
                 custom_dimensions: dict,
                 app_insights_connection_string: str,
                 worker_id: ContextVar,
                 mini_batch_id: ContextVar,
                 quota_audience: ContextVar,
                 batch_pool: ContextVar):
        """Init method."""
        self.__custom_dimensions = custom_dimensions
        self.__worker_id = worker_id
        self.__mini_batch_id = mini_batch_id
        self.__quota_audience = quota_audience
        self.__batch_pool = batch_pool
        self.__logger = logging.getLogger("BatchComponentMetricsLogger")
        self.__logger.propagate = False
        self.__logger.setLevel(logging.INFO)

        azure_handler = AzureEventHandler(connection_string=app_insights_connection_string)
        self.__logger.addHandler(azure_handler)

    def _common_custom_dimensions(self, custom_dimensions):
        custom_dimensions["MiniBatchId"] = self.__mini_batch_id.get()
        custom_dimensions["WorkerId"] = self.__worker_id.get()
        custom_dimensions["QuotaAudience"] = self.__quota_audience.get()
        custom_dimensions["BatchPool"] = self.__batch_pool.get()

    def emit_request_completed(
        self,
        latency: int,
        request_internal_id: str,
        client_request_id: str,
        endpoint_uri: str = "",
        status_code: str = "0",
        model_response_code: str = "",
        client_exception: str = "",
        is_retriable: bool = False
    ):
        """Emit request completed for AppInsightsEventsClient."""
        custom_dimensions = self.__custom_dimensions.copy()
        self._common_custom_dimensions(custom_dimensions=custom_dimensions)

        custom_dimensions["RequestInternalId"] = request_internal_id
        custom_dimensions["ClientRequestId"] = client_request_id
        custom_dimensions["IsRetriable"] = is_retriable
        custom_dimensions["EndpointUri"] = endpoint_uri
        custom_dimensions["StatusCode"] = status_code
        custom_dimensions["ModelResponseCode"] = model_response_code
        custom_dimensions["ClientException"] = client_exception
        custom_dimensions["LatencyMilliseconds"] = latency

        extra = {
            "custom_dimensions": custom_dimensions
        }
        self.__logger.info("request_completed", extra=extra)

    def emit_tokens_generated(
        self,
        generated_tokens: int,
        context_tokens: int,
        endpoint_uri: str = "",
    ):
        """Emit tokens generated for AppInsightsEventsClient."""
        custom_dimensions = self.__custom_dimensions.copy()
        self._common_custom_dimensions(custom_dimensions=custom_dimensions)

        custom_dimensions["EndpointUri"] = endpoint_uri
        custom_dimensions["TokensGenerated"] = generated_tokens
        custom_dimensions["ContextTokenSize"] = context_tokens

        extra = {
            "custom_dimensions": custom_dimensions
        }
        self.__logger.info("tokens_generated", extra=extra)

    def emit_request_concurrency(
        self,
        endpoint_uri: str,
        active_requests: int,
        quota_reserved: int
    ):
        """Emit request concurrency for AppInsightsEventsClient."""
        custom_dimensions = self.__custom_dimensions.copy()
        self._common_custom_dimensions(custom_dimensions=custom_dimensions)

        custom_dimensions["EndpointUri"] = endpoint_uri
        custom_dimensions["ActiveRequests"] = active_requests
        custom_dimensions["QuotaReserved"] = quota_reserved

        extra = {
            "custom_dimensions": custom_dimensions
        }
        self.__logger.info("request_concurrency", extra=extra)

    def emit_worker_concurrency(
        self,
        worker_concurrency: int
    ):
        """Emit worker concurrency for AppInsightsEventsClient."""
        custom_dimensions = self.__custom_dimensions.copy()
        self._common_custom_dimensions(custom_dimensions=custom_dimensions)

        custom_dimensions["WorkerConcurrency"] = worker_concurrency

        extra = {
            "custom_dimensions": custom_dimensions
        }
        self.__logger.info("worker_concurrency", extra=extra)

    def emit_row_completed(
        self,
        row_count: int,
        result: str = "SUCCESS"
    ):
        """Emit row completed for AppInsightsEventsClient."""
        custom_dimensions = self.__custom_dimensions.copy()
        self._common_custom_dimensions(custom_dimensions=custom_dimensions)

        custom_dimensions["Result"] = result
        custom_dimensions["RowCount"] = row_count

        extra = {
            "custom_dimensions": custom_dimensions
        }

        self.__logger.info("rows_completed", extra=extra)

    def emit_quota_operation(self, operation: str, status_code: int, lease_id: str, amount: int):
        """Emit quota operation for AppInsightsEventsClient."""
        custom_dimensions = self.__custom_dimensions.copy()
        self._common_custom_dimensions(custom_dimensions=custom_dimensions)

        custom_dimensions["LeaseId"] = lease_id
        custom_dimensions["Operation"] = operation
        custom_dimensions["QuotaAmount"] = amount
        custom_dimensions["StatusCode"] = status_code

        extra = {
            "custom_dimensions": custom_dimensions
        }

        self.__logger.info("quota_operation", extra=extra)

    def emit_mini_batch_started(
        self,
        input_row_count: int
    ):
        """Emit mini batch started for AppInsightsEventsClient."""
        custom_dimensions = self.__custom_dimensions.copy()
        self._common_custom_dimensions(custom_dimensions=custom_dimensions)

        custom_dimensions["InputRowCount"] = input_row_count

        extra = {
            "custom_dimensions": custom_dimensions
        }

        self.__logger.info("mini_batch_started", extra=extra)

    def emit_mini_batch_completed(
        self,
        input_row_count: int,
        output_row_count: int,
        exception: str = None,
        stacktrace: str = None
    ):
        """Emit mini batch completed for AppInsightsEventsClient."""
        custom_dimensions = self.__custom_dimensions.copy()
        self._common_custom_dimensions(custom_dimensions=custom_dimensions)

        custom_dimensions["InputRowCount"] = input_row_count
        custom_dimensions["OutputRowCount"] = output_row_count
        custom_dimensions["Exception"] = exception
        custom_dimensions["StackTrace"] = stacktrace

        extra = {
            "custom_dimensions": custom_dimensions
        }

        self.__logger.info("mini_batch_completed", extra=extra)

    def emit_batch_driver_init(
        self,
        job_params: dict,
    ):
        """Emit batch driver init for AppInsightsEventsClient."""
        custom_dimensions = self.__custom_dimensions.copy()
        self._common_custom_dimensions(custom_dimensions=custom_dimensions)

        if isinstance(job_params, dict):
            custom_dimensions["JobParameters"] = json.dumps(job_params)

        extra = {
            "custom_dimensions": custom_dimensions
        }

        self.__logger.info("batch_driver_init", extra=extra)

    def emit_batch_driver_shutdown(
        self,
        job_params: dict,
    ):
        """Emit batch driver shutdown for AppInsightsEventsClient."""
        custom_dimensions = self.__custom_dimensions.copy()
        self._common_custom_dimensions(custom_dimensions=custom_dimensions)

        if isinstance(job_params, dict):
            custom_dimensions["JobParameters"] = json.dumps(job_params)

        extra = {
            "custom_dimensions": custom_dimensions
        }

        self.__logger.info("batch_driver_shutdown", extra=extra)
