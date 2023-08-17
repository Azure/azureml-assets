import datetime
import json
import requests

from azure.ai.ml.identity import AzureMLOnBehalfOfCredential, CredentialUnavailableError
from pyspark.sql import Row
from typing import List

from shared_utilities.log_utils import log_utils

def publish_metric(
        signal_metrics: List[Row],
        monitor_name: str,
        signal_name: str,
        data_window_start: datetime,
        data_window_end: datetime,
        location: str,
        ws_resource_uri: str):
    """Publish the signal metrics."""
    log_utils.info(f"Attempting to publish metrics of monitor {monitor_name}, signal {signal_name}.")

    ws_resource_uri = ws_resource_uri.strip("/")
    metrics_url = f"https://{location}.monitoring.azure.com/{ws_resource_uri}/metrics" # TODO: use schedule subresource instead?

    log_utils.info(f"Publishing metrics to Azure resource with url {metrics_url}.")

    succeeded_count = 0
    failed_count = 0
    for metric in signal_metrics:
        payload = to_metric_payload(metric, monitor_name, signal_name, data_window_start, data_window_end)

        try:
            response = __post_metric(payload, metrics_url)
            response.raise_for_status()
            succeeded_count += 1
        except requests.exceptions.RequestException as err:
            log_utils.warning(f"Failed to publish a metric. Error: {str(err)}.")
            failed_count += 1

    total_count = succeeded_count + failed_count
    log_utils.info(f"Published Azure monitor metrics for monitor {monitor_name}, signal {signal_name}:"\
                   + f"Total requested: {total_count}, success: {succeeded_count}, failure: {failed_count}")

def to_metric_payload(
        metric: Row,
        monitor_name: str,
        signal_name: str,
        data_window_start: datetime,
        data_window_end: datetime) -> dict:
    """Convert to a dictionary object for metric output."""
    try:
        metric_name = metric["metric_name"]
        metric_value = metric["metric_value"]
        group = metric["group"]
    except KeyError as err:
        log_utils.error(f"A required column is missing from the metric. Error: {str(err)}, metric: {json.dumps(metric.asDict)}.")
        raise

    group_dimension = metric["group_dimension"] if "group_dimension" in metric else None

    payload = {
        "time": datetime.utcnow().isoformat(" "),
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
                            data_window_start.isoformat(" "),
                            data_window_end.isoformat(" "),
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
        log_utils.error(f"Unable to get an AML OBO token, error: {str(err)}.")
        raise

    headers = {
        "Authorization": f"Bearer {auth_token}"
    }
    response = requests.post(url, json = payload, headers=headers)

    return response
