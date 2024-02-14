# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Data Collector Data Window Component."""

import argparse

from pyspark.sql import DataFrame
from pyspark.sql.functions import posexplode, concat_ws, udf, from_json, when, col, to_json
from pyspark.sql.types import StringType, AtomicType
from py4j.protocol import Py4JJavaError
from dateutil import parser
from datetime import datetime
from shared_utilities.momo_exceptions import DataNotFoundError
from shared_utilities.io_utils import init_spark, save_spark_df_as_mltable
from shared_utilities.event_utils import add_tags_to_root_run
from shared_utilities.constants import (
    MDC_CORRELATION_ID_COLUMN, MDC_DATA_COLUMN, MDC_DATAREF_COLUMN, AML_MOMO_ERROR_TAG
)
from mdc_preprocessor_helper import (
    get_file_list, set_data_access_config, serialize_credential, copy_appendblob_to_blockblob
)
from model_data_collector_preprocessor.store_url import StoreUrl
from model_data_collector_preprocessor.mdc_preprocessor_helper import deserialize_credential
# from store_url import StoreUrl


def _mdc_uri_folder_to_raw_spark_df(start_datetime: datetime, end_datetime: datetime, store_url: StoreUrl,
                                    add_tags_func=None) -> DataFrame:
    """Read raw MDC data, and return in a Spark DataFrame."""
    def handle_data_not_found():
        add_tags_func({AML_MOMO_ERROR_TAG: "No data found for the given time window."})
        raise DataNotFoundError(
            f"No data found for the given time window: {start_datetime} to {end_datetime} "
            f"in input {store_url.get_hdfs_url()}. "
            "We expect folder pattern <root>/YYYY/MM/DD/HH/<your_log>.jsonl")

    add_tags_func = add_tags_func or add_tags_to_root_run

    try:
        return _uri_folder_to_spark_df(start_datetime, end_datetime, store_url, soft_delete_enabled=False)
    except FileNotFoundError:
        handle_data_not_found()
    except Py4JJavaError as pe:
        if "This endpoint does not support BlobStorageEvents or SoftDelete" in pe.java_exception.getMessage():
            blockblob_url = copy_appendblob_to_blockblob(store_url, start_datetime, end_datetime)
            return _uri_folder_to_spark_df(start_datetime, end_datetime, blockblob_url, soft_delete_enabled=True)
        else:
            raise pe


def _uri_folder_to_spark_df(start_datetime: datetime, end_datetime: datetime, store_url: StoreUrl,
                            soft_delete_enabled: bool = False) -> DataFrame:
    scheme = "azureml" if soft_delete_enabled else "abfs"
    file_list = get_file_list(start_datetime, end_datetime, store_url=store_url, scheme=scheme)
    if not file_list:
        raise FileNotFoundError(f"No data found for the given time window: {start_datetime} to {end_datetime}")
    # print("DEBUG file_list:", file_list)

    spark = init_spark()
    if not soft_delete_enabled:
        # need to set credential in spark conf for abfs access
        set_data_access_config(spark, store_url=store_url)
    df = spark.read.json(file_list)
    if df.rdd.isEmpty():
        raise FileNotFoundError(f"No data found for the given time window: {start_datetime} to {end_datetime}")

    return df


def _load_dataref_into_data_column(df: DataFrame, store_url: StoreUrl) -> DataFrame:
    if MDC_DATAREF_COLUMN not in df.columns:
        return df  # no dataref column, return df directly
    if MDC_DATA_COLUMN not in df.columns:
        # TODO need full schema override
        raise NotImplementedError(f"{MDC_DATAREF_COLUMN} column without {MDC_DATA_COLUMN} is not supported yet.")

    @udf(returnType=StringType())
    def load_json_str(path):
        if not path:
            return None

        store_url = StoreUrl(path)
        credential = deserialize_credential(broadcasted_credential.value)
        return store_url.read_file_content(credential=credential)

    spark = init_spark()
    serialized_credential = serialize_credential(store_url.get_credential())
    broadcasted_credential = spark.sparkContext.broadcast(serialized_credential)

    # TODO separate df into 2 parts, one with dataref, one without, then only handle the one with dataref, and
    #      union the 2 parts back together

    # load json string from dataref to ref_json_str column
    df = df.withColumn("ref_json_str", load_json_str(df[MDC_DATAREF_COLUMN]))
    # parse json string to json object with from_json(), into ref_data column
    df = df.withColumn("ref_data", from_json(df["ref_json_str"], df.schema[MDC_DATA_COLUMN].dataType))
    # replace data with ref_data if data column is null
    df = df.withColumn(MDC_DATA_COLUMN,
                       when(col(MDC_DATA_COLUMN).isNull(), col("ref_data")).otherwise(col(MDC_DATA_COLUMN)))
    # drop temp columns
    df = df.drop("ref_json_str", "ref_data")

    return df


def _extract_data_and_correlation_id(df: DataFrame, extract_correlation_id: bool) -> DataFrame:
    """Extract data and correlation id from the MDC logs."""
    columns = [MDC_DATA_COLUMN]
    if extract_correlation_id:
        columns.append(MDC_CORRELATION_ID_COLUMN)
        # explode the data column of array type to multiple rows with index
        df = df[columns].select(posexplode(MDC_DATA_COLUMN).alias("index", "value"), MDC_CORRELATION_ID_COLUMN)
        # set the new correlationid as {correlationid}_{index}
        df = df.withColumn(MDC_CORRELATION_ID_COLUMN, concat_ws("_", MDC_CORRELATION_ID_COLUMN, "index")).drop("index")
        # select the 1st level features as columns
        df = df.select("value.*", MDC_CORRELATION_ID_COLUMN)
    else:
        df = df[columns].select(posexplode(MDC_DATA_COLUMN).alias("index", "value"))
        df = df.select("value.*")
    return df


def _convert_complex_columns_to_json_string(df: DataFrame) -> DataFrame:
    for col_schema in df.schema:
        if not isinstance(col_schema.dataType, AtomicType):
            df = df.withColumn(col_schema.name, to_json(df[col_schema.name]))
    return df


def _mdc_uri_folder_to_preprocessed_spark_df(
        data_window_start: str, data_window_end: str, store_url: StoreUrl, extract_correlation_id: bool,
        add_tags_func=None) -> DataFrame:
    """Read raw MDC data, preprocess, and return in a Spark DataFrame."""
    # Parse the dates
    start_datetime = parser.parse(data_window_start)
    end_datetime = parser.parse(data_window_end)

    df = _mdc_uri_folder_to_raw_spark_df(start_datetime, end_datetime, store_url, add_tags_func)
    print("df converted from MDC raw uri folder:")
    df.select("data").show(truncate=False)
    df.printSchema()

    df = _load_dataref_into_data_column(df, store_url)
    df.select("data").show(truncate=False)
    df.printSchema()

    df = _extract_data_and_correlation_id(df, extract_correlation_id)
    df.show()
    df.printSchema()

    return df


def mdc_preprocessor(
        data_window_start: str,
        data_window_end: str,
        input_data: str,
        preprocessed_input_data: str,
        extract_correlation_id: bool):
    """Extract data based on window size provided and preprocess it into MLTable.

    Args:
        data_window_start: The start date of the data window.
        data_window_end: The end date of the data window.
        input_data: The data asset on which the date filter is applied.
        preprocessed_data: The mltable path pointing to location where the outputted mltable will be written to.
        extract_correlation_id: The boolean to extract correlation Id from the MDC logs.
    """
    store_url = StoreUrl(input_data)
    transformed_df = _mdc_uri_folder_to_preprocessed_spark_df(data_window_start, data_window_end, store_url,
                                                              extract_correlation_id)
    # TODO remove this step after we switch our interface from mltable to uri_folder
    transformed_df = _convert_complex_columns_to_json_string(transformed_df)

    save_spark_df_as_mltable(transformed_df, preprocessed_input_data)


def run():
    """Compute data window and preprocess data from MDC."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_window_start", type=str)
    parser.add_argument("--data_window_end", type=str)
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--extract_correlation_id", type=str)
    parser.add_argument("--preprocessed_input_data", type=str)
    args = parser.parse_args()

    mdc_preprocessor(
        args.data_window_start,
        args.data_window_end,
        args.input_data,
        args.preprocessed_input_data,
        eval(args.extract_correlation_id.capitalize()),
    )


if __name__ == "__main__":
    run()
