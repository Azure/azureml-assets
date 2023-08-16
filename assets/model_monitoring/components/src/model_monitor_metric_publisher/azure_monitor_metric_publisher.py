import datetime
import json
import requests

from pyspark.sql import Row
from typing import List

from shared_utilities.log_utils import log_utils

def publish_metric(
        signal_metrics: List[Row],
        monitor_name: str,
        signal_name: str,
        location: str,
        ws_resource_uri: str):
    """Publish the signal metrics."""
    log_utils.info(f"Attempting to publish metrics of monitor {monitor_name}, signal {signal_name}.")

    ws_resource_uri = ws_resource_uri.strip("/")
    metrics_url = f"https://{location}.monitoring.azure.com/{ws_resource_uri}/metrics"

    log_utils.info(f"Publishing metrics to Azure resource with url {metrics_url}.")

    succeeded_count = 0
    failed_count = 0
    for metric in signal_metrics:
        payload = __to_metric_payload(metric, monitor_name, signal_name)

        try:
            response = __post_metric(payload, metrics_url)
            response.raise_for_status()
            succeeded_count += 1
        except requests.exceptions.RequestException as err:
            log_utils.warning(f"Failed to publish a metric. Error: {str(err)}")
            failed_count += 1

    total_count = succeeded_count + failed_count
    log_utils.info(f"Published Azure monitor metrics for monitor {monitor_name}, signal {signal_name}:"\
                   + f"Total requested: {total_count}, success: {succeeded_count}, failure: {failed_count}")

def __to_metric_payload(metric: Row, monitor_name: str, signal_name: str) -> dict:
    """Convert to a dictionary object for metric output."""
    group = metric["group"]
    metric_name = metric["metric_name"]
    metric_value = metric["metric_value"]
    threshold_value = metric["threshold_value"]

    if "custom_dimensions" in metric:
        custom_dims_str = json.dumps(metric["custom_dimensions"])
    else:
        custom_dims_str = ""

    if "group_dimension" in metric:
        group_dimension = metric["group_dimension"]
        group_dim_str = f"{group}.{group_dimension}"
    else:
        group_dim_str = group

    payload = {
        "time": datetime.utcnow().isoformat(" "),
        "data": {
            "baseData": {
                "metric": metric_name,
                "namespace": f"ModelMonitor.{monitor_name}",
                "dimNames": [
                    "CustomDimensions",
                    "GroupDimension",
                    "MonitorName",
                    "SignalName",
                    "ThresholdValue",
                ],
                "series": [
                    {
                        "dimValues": [
                            custom_dims_str,
                            group_dim_str,
                            monitor_name,
                            signal_name,
                            threshold_value,
                        ],
                        "min": metric_value,
                        "max": metric_value,
                        "sum": metric_value,
                        "count": 1,
                    }
                ]
            },
        }
    }
    return payload


def __post_metric(payload: dict, url: str) -> requests.Response:
    """Make a request to Azure monitor to publish a metric. Return the reponse object."""
    # todo
    # credential = AzureMLOnBehalfOfCredential()
    auth_token = "dummy" # auth_token = credential.get_token('https://monitor.azure.com') # todo: try - catch

    headers = {
        "Authorization": f"Bearer {auth_token}"
    }
    response = requests.post(url, json = payload, headers=headers)

    return response
