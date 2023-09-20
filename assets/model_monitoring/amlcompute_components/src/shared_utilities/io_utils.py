# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to read write data."""

import numpy as np
from pyspark.sql import SparkSession


def _get_datastore_id():
    import os
    import re
    workspace_scope = os.environ['AZUREML_WORKSPACE_SCOPE'].lower()
    pattern = r"^/subscriptions/(.+?)/resourcegroups/(.+?)/providers"\
              + r"/microsoft.machinelearningservices/workspaces/(.+?)$"

    match = re.search(pattern, workspace_scope)
    subscription_id = match.group(1)
    resource_group_name = match.group(2)
    workspace_name = match.group(3)

    return f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group_name}"\
           + f"/workspaces/{workspace_name}/datastores/workspaceblobstore"


def _get_datastore_relative_data_path(data_path):
    import os
    import re
    job_id = os.environ['AZUREML_RUN_ID']
    pattern = r"^(.+?)/cap/data-capability/wd/(.+?)$"

    match = re.search(pattern, data_path)
    output_folder = match.group(2)

    return f"paths/azureml/{job_id}/{output_folder}"


def convert_to_azureml_uri(data_path: str):
    import os
    datastore_id = _get_datastore_id()
    relative_data_path = _get_datastore_relative_data_path(data_path)

    return os.path.join(datastore_id, relative_data_path)


def init_spark():
    """Get or create spark session."""
    spark = SparkSession.builder.appName("AccessParquetFiles").getOrCreate()
    return spark


def np_encoder(object):
    """Json encoder for numpy types."""
    if isinstance(object, np.generic):
        return object.item()


def read_mltable_in_spark(mltable_path: str):
    """Read mltable in spark."""
    spark = init_spark()
    df = spark.read.mltable(mltable_path)
    return df


def save_spark_df_as_mltable(metrics_df, folder_path: str):
    """Save spark dataframe as mltable."""
    metrics_df.write.option("output_format", "parquet").option(
        "overwrite", True
    ).mltable(f"file://{folder_path}")
