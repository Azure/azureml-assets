# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contains functionality for publishing run metrics."""
import mlflow
import os


def _get_experiment_id():
    return os.environ.get("MLFLOW_EXPERIMENT_ID")


def _create_filter_query(
    monitor_name: str, signal_name: str, feature_name: str, metric_name: str
):
    def _generate_filter(tag: str, value: str):
        return f"tags.azureml.{tag} = '{value}'"

    filters = []
    if monitor_name is not None:
        filters.append(_generate_filter("monitor_name", monitor_name))
    if signal_name is not None:
        filters.append(_generate_filter("signal_name", signal_name))
    if feature_name is not None:
        filters.append(_generate_filter("feature_name", feature_name))
    if metric_name is not None:
        filters.append(_generate_filter("metric_name", metric_name))
    return " and ".join(filters)


def _create_run_tags(
    monitor_name: str, signal_name: str, feature_name: str, metric_name: str
):
    tags = {}
    if monitor_name is not None:
        tags["azureml.monitor_name"] = monitor_name
    if signal_name is not None:
        tags["azureml.signal_name"] = signal_name
    if feature_name is not None:
        tags["azureml.feature_name"] = feature_name
    if metric_name is not None:
        tags["azureml.metric_name"] = metric_name
    return tags


def get_or_create_run_id(
    monitor_name: str, signal_name: str, feature_name: str, metric_name: str
) -> str:
    """Get or create a run id for a given monitor, signal, feature, and metric."""
    experiment_id = _get_experiment_id()
    if experiment_id is None:
        print("No experiment id found. Skipping publishing run metrics.")
        return None
    filter_query = _create_filter_query(
        monitor_name, signal_name, feature_name, metric_name
    )
    print(f"Fetching run metric with filter: {filter_query}")
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_query,
        order_by=["start_time"],
    )
    if len(runs) == 0:

        run_id = (
            mlflow.client.MlflowClient()
            .create_run(
                experiment_id=experiment_id,
                tags=_create_run_tags(
                    monitor_name, signal_name, feature_name, metric_name
                ),
            )
            .info.run_id
        )
        print(f"Created run metric with id '{run_id}'.")
    else:
        run_id = runs.iloc[0].run_id
        print(f"Fetching run metric with id '{run_id}'")
    return run_id


def publish_metric(run_id: str, value: float, threshold, step: int):
    """Publish a metric to the run metrics store."""
    metrics = {}
    metrics["value"] = value
    if threshold is not None:
        metrics["threshold"] = float(threshold)
    publish_metrics(run_id=run_id, metrics=metrics, step=step)


def publish_metrics(run_id: str, metrics: dict, step: int):
    """Publish metrics to the run metrics store."""
    print(f"Publishing metrics to run id '{run_id}'.")
    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_metrics(metrics=metrics, step=step)
