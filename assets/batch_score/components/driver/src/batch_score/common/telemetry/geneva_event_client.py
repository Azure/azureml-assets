# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Geneva client to emit telemetry events."""

import importlib
import logging
import platform
import sys
import uuid

from .events.batch_score_event import BatchScoreEvent
from .required_fields import RequiredFields
from .standard_fields import AzureMLTelemetryComputeType, AzureMLTelemetryOS, StandardFields

COMPONENT_NAME = "BatchScoreLlm"


class GenevaEventClient():
    """Geneva event client."""

    def __init__(self):
        """Initialize GenevaEventClient."""
        self._logger = logging.getLogger("GenevaEventClientLogger")
        self._logger.setLevel(logging.DEBUG)

        try:
            telemetry_module = importlib.import_module("azureml_common.parallel_run.telemetry_logger")
            self._event_logger = getattr(telemetry_module, "log_message_internal_v2")
            self._logger.debug("Sucessfully imported log_message_internal_v2.")
        except Exception as e:
            msg = f"Failed to import log_message_internal_v2, exception: {e}."
            self._logger.error(msg)
            raise ImportError(msg)

    def emit_event(self, event: BatchScoreEvent):
        """Emit batch score telemetry event."""
        try:
            self._event_logger(
                level="Information",
                log_locally=False,
                required_fields=self.generate_required_fields(event),
                standard_fields=self.generate_standard_fields(event),
                extension_fields=self.generate_extension_fields(event),
                run_id=event.run_id)
        except Exception as e:
            self._logger.error(f"Failed to emit events using log_message_internal_v2, exception: {e}.")

    def generate_required_fields(self, event: BatchScoreEvent):
        """Generate required fields (Part A) for any batch score events."""
        required_fields = RequiredFields(
            subscriptionId=event.subscription_id,
            workspaceId=event.workspace_id,
            correlationId=str(uuid.uuid4()),
            componentName=COMPONENT_NAME,
            eventName=event.name,
        )
        return required_fields

    def generate_standard_fields(self, event: BatchScoreEvent):
        """Generate standard fields (Part B) for any batch score events."""
        standard_fields = StandardFields(
            Attribution="Other",
            WorkspaceRegion=event.workspace_location,
            ComputeType=AzureMLTelemetryComputeType.current(),
            ClientVersion=event.component_version,
            ClientOS=self._get_client_os(sys.platform),
            ClientOSVersion=platform.platform(),
            RunId=event.run_id,
            ParentRunId=event.parent_run_id,
            ExperimentId=event.experiment_id,
        )
        return standard_fields

    def generate_extension_fields(self, event: BatchScoreEvent):
        """Generate extension fields (Part C) as a dictionary for any batch score events."""
        """Fields exist in Part A or B are removed."""
        # Convert all the values to string to avoid JSON serialization errors when sending the event to Geneva.
        extension_fields = {k: str(v) for k, v in event.to_dictionary(include_empty_keys=False).items() if k not in [
            'subscription_id',
            'workspace_id',
            'workspace_location',
            'run_id',
            'parent_run_id',
            'experiment_id',
        ]}

        extension_fields['api_type'] = str(event.api_type)
        extension_fields['authentication_type'] = str(event.authentication_type)
        extension_fields['endpoint_type'] = str(event.endpoint_type)
        extension_fields['event_time'] = event.event_time.isoformat()

        return extension_fields

    def _get_client_os(self, client_os):
        """Parse the client OS type from sys.platform() into AML schema."""
        if client_os in ["win32", "cygwin"]:
            return int(AzureMLTelemetryOS.Windows)
        if client_os == "linux":
            return int(AzureMLTelemetryOS.Linux)
        if client_os == "darwin":
            return int(AzureMLTelemetryOS.MacOS)
        return int(AzureMLTelemetryOS.Others)
