# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module to record appinsight events."""

import os
import logging
import sys
import time

from opencensus.ext.azure.log_exporter import AzureLogHandler


class AppInsightsRecorder:
    """Class to initiate appinsights."""

    ENV_API_KEY = "AML_APP_INSIGHTS_KEY"

    tc_singleton = None

    """Batching parameters, whichever of the below conditions gets hit first will trigger a send.
        send_interval: interval in seconds
        send_buffer_size: max number of items to batch before sending
    """
    send_interval = 5.0
    send_buffer_size = 100

    def __init__(self):
        """Initiate class variables."""
        if AppInsightsRecorder.tc_singleton is None:
            try:
                app_insights_key = os.environ[AppInsightsRecorder.ENV_API_KEY]
                logger = logging.getLogger(__name__)
                logger.setLevel("INFO")
                azureLogHandler = AzureLogHandler(
                    instrumentation_key=app_insights_key,
                    export_interval=AppInsightsRecorder.send_interval,
                    max_batch_size=AppInsightsRecorder.send_buffer_size,
                )
                logger.addHandler(azureLogHandler)
                AppInsightsRecorder.tc_singleton = logger
            except Exception as exception:
                logger.error(f"Exception occurred while processing {exception}")
                logging.exception(
                    "Failed to initialize Application Insights Client.\n"
                    "Check that there is a valid Instrumentation Key in {0}".format(AppInsightsRecorder.ENV_API_KEY)
                )
                sys.exit(1)

    def on_receive(self, raw_msg):
        """Publish received message as events."""
        # The message is populated with the following four system properties.
        # See rsyslog.conf files for the template from syslog
        (fd, host, time, unparsed_msg) = raw_msg.split(",", 3)
        # The original content is prefixed and comma-separated with the request id (if one exists) at the front.
        (request_id, message) = unparsed_msg.split(",", 1)
        workspace_name = os.environ.get("WORKSPACE_NAME", "")
        service_name = os.environ.get("SERVICE_NAME", "")
        properties = {
            "custom_dimensions": {
                "Container Id": host,
                "Timestamp": time,
                "Content": message,
                "Request Id": request_id,
                "Workspace Name": workspace_name,
                "Service Name": service_name,
            }
        }
        AppInsightsRecorder.tc_singleton.info(fd, extra=properties)

    def on_exit(self):
        """Clean resources on exit."""
        # The following enables the context to switch to the sender thread.
        # It can send the remaining messages before shutting down.
        time.sleep(1)
        sys.stdout.flush()
