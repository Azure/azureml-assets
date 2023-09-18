# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to read write data."""

import numpy as np
from pyspark.sql import SparkSession


def init_spark():
    """Get or create spark session."""
    spark = SparkSession.builder.appName("AccessParquetFiles").getOrCreate()
    return spark


def read_mltable_in_spark(mltable_path: str):
    """Read mltable in spark."""
    spark = init_spark()
    df = spark.read.mltable(mltable_path)
    return df

"""
azureml://subscriptions/<subscription-id>/
    resourcegroups/<resourcegroup-name>/workspaces/<workspace-name>/
    datastores/<datastore-name>
"""
def save_spark_df_as_mltable_from_command_jobs(metrics_df, folder_path: str):
    """Save spark dataframe as mltable from command jobs."""
    # workspace_scope_var = os.environ['AZUREML_WORKSPACE_SCOPE']
    # workspace_scope = workspace_scope_var.replace("/subscriptions", "subscriptions").replace("/providers/Microsoft.MachineLearningServices", "")
    # datastore_id = os.path.join("azureml://", workspace_scope, "datastore/workspaceblobstore")
    # job_id = os.environ['MLFLOW_RUN_ID']
    # output_folder = os.path.split(folder_path)[1]
    # output_path = os.path.join(datastore_id, "paths/azureml", job_id, output_folder)
    # print(f"save_spark_df_as_mltable_from_command_jobs: workspace_scope_var: {workspace_scope_var}, workspace_scope: {workspace_scope}, folder_path: {folder_path}, datastore_id: {datastore_id}, output_path: {output_path}")
    
    save_spark_df_as_mltable(metrics_df, output_path)

def save_spark_df_as_mltable(metrics_df, folder_path: str):
    """Save spark dataframe as mltable."""
    metrics_df.write.option("output_format", "parquet").option(
        "overwrite", True
    ).mltable(f"file://{folder_path}")


def np_encoder(object):
    """Json encoder for numpy types."""
    if isinstance(object, np.generic):
        return object.item()
