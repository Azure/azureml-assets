# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper methods to publish signal metrics to Azure Monitor of users."""

from datetime import datetime, timezone
import json
import requests

from azure.ai.ml.identity import AzureMLOnBehalfOfCredential, CredentialUnavailableError
from pyspark.sql import Row
from typing import List

from shared_utilities.log_utils import log_info, log_warning, log_error

def publish_metric(
        metrics: List[Row],
        monitor_name: str,
        signal_name: str,
        window_start: datetime,
        window_end: datetime,
        location: str,
        ws_resource_uri: str):
    """Publish the computed metrics to the specified Azure resource."""
    log_info(f"Attempting to publish metrics of monitor {monitor_name}, signal {signal_name}.")

    ws_resource_uri = ws_resource_uri.strip("/")
    metrics_url = f"https://{location}.monitoring.azure.com/{ws_resource_uri}/metrics" # TODO: use schedule subresource instead?

    log_info(f"Publishing metrics to Azure resource with url {metrics_url}.")

    succeeded_count = 0
    failed_count = 0
    for metric in metrics:
        payload = to_metric_payload(metric, monitor_name, signal_name, window_start, window_end)

        try:
            response = __post_metric(payload, metrics_url)
            response.raise_for_status()
            succeeded_count += 1
        except requests.exceptions.RequestException as err:
            log_warning(f"Failed to publish a metric. Error: {str(err)}.")
            failed_count += 1

    total_count = succeeded_count + failed_count
    log_info(f"Published Azure monitor metrics for monitor {monitor_name}, signal {signal_name}:"\
             + f"Total requested: {total_count}, success: {succeeded_count}, failure: {failed_count}")

def to_metric_payload(
        metric: Row,
        monitor_name: str,
        signal_name: str,
        window_start: datetime,
        window_end: datetime) -> dict:
    """Convert a computed metric row to a dictionary object."""
    try:
        metric_name = metric["metric_name"]
        metric_value = metric["metric_value"]
        group = metric["group"]
    except KeyError as err:
        log_error(f"A required column is missing from the metric. Error: {str(err)}, metric: {json.dumps(metric.asDict)}.")
        raise

    group_dimension = metric["group_dimension"] if "group_dimension" in metric else None

    payload = {
        "time": datetime.now(timezone.utc).isoformat(" "),
        "data": {
            "baseData": {
                "metric": metric_name,
                "namespace": "ModelMonitor",
                "dimNames": [
                    "Group"
                    "GroupDimension",
                    "MonitorName",
                    "SignalName",
                    "DataWindowStart",
                    "DataWindowEnd",
                ],
                "series": [
                    {
                        "dimValues": [
                            group,
                            group_dimension,
                            monitor_name,
                            signal_name,
                            window_start.isoformat(" "),
                            window_end.isoformat(" "),
                        ],
                        "min": metric_value,
                        "max": metric_value,
                        "sum": metric_value,
                        "count": 1,
                    }
                ]
            }
        }
    }
    return payload


def __post_metric(payload: dict, url: str) -> requests.Response:
    """Make a request to Azure monitor to publish a metric. Return the reponse object."""
    credential = AzureMLOnBehalfOfCredential()

    try:
        auth_token = credential.get_token('https://monitor.azure.com')
    except CredentialUnavailableError as err:
        log_error(f"Unable to get an AML OBO token, error: {str(err)}.")
        raise

    headers = {
        "Authorization": f"Bearer {auth_token}"
    }
    response = requests.post(url, json = payload, headers=headers)

    return response
