# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared utility methods for preprocessor Components."""

from pyspark.sql import DataFrame
from pyspark.sql.functions import posexplode, concat_ws, udf, from_json, when, col, to_json
from pyspark.sql.types import StringType, AtomicType
from py4j.protocol import Py4JJavaError
from dateutil import parser
from datetime import datetime
from shared_utilities.momo_exceptions import DataNotFoundError, InvalidInputError
from shared_utilities.io_utils import init_spark
from shared_utilities.event_utils import add_tags_to_root_run
from shared_utilities.constants import (
    MDC_CORRELATION_ID_COLUMN, MDC_DATA_COLUMN, MDC_DATAREF_COLUMN, AML_MOMO_ERROR_TAG
)

from model_data_collector_preprocessor.mdc_preprocessor_helper import (
    get_file_list, set_data_access_config, serialize_credential, copy_appendblob_to_blockblob
)
from shared_utilities.store_url import StoreUrl
from model_data_collector_preprocessor.mdc_preprocessor_helper import deserialize_credential


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
        try:
            return _uri_folder_to_spark_df(start_datetime, end_datetime, store_url, soft_delete_enabled=False)
        except Py4JJavaError as pe:
            if "This endpoint does not support BlobStorageEvents or SoftDelete" in pe.java_exception.getMessage():
                blockblob_url = copy_appendblob_to_blockblob(store_url, start_datetime, end_datetime)
                return _uri_folder_to_spark_df(start_datetime, end_datetime, blockblob_url, soft_delete_enabled=True)
            else:
                raise pe
    except DataNotFoundError:
        handle_data_not_found()


def _uri_folder_to_spark_df(start_datetime: datetime, end_datetime: datetime, store_url: StoreUrl,
                            soft_delete_enabled: bool = False) -> DataFrame:
    scheme = "azureml" if soft_delete_enabled else "abfs"
    file_list = get_file_list(start_datetime, end_datetime, store_url=store_url, scheme=scheme)
    if not file_list:
        raise DataNotFoundError(f"No data found for the given time window: {start_datetime} to {end_datetime}")
    # print("DEBUG file_list:", file_list)

    spark = init_spark()
    if not soft_delete_enabled:
        # need to set credential in spark conf for abfs access
        set_data_access_config(spark, store_url=store_url)
    df = spark.read.json(file_list)
    if df.rdd.isEmpty():
        raise DataNotFoundError(f"No data found for the given time window: {start_datetime} to {end_datetime}")

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
    df.select("data").show()
    df.printSchema()

    df = _load_dataref_into_data_column(df, store_url)
    df.select("data").show()
    df.printSchema()

    df = _extract_data_and_correlation_id(df, extract_correlation_id)
    df.show()
    df.printSchema()

    return df


def _filter_df_by_time_window(df: DataFrame, data_window_start: datetime, data_window_end: datetime) -> DataFrame:
    """Filter dataframe on its end_time column to fit within the given data window: [start, end)."""
    df = df.filter(df.end_time >= data_window_start).filter(df.end_time < data_window_end)
    return df


def _count_dropped_rows_with_error(
        original_df_row_count: int,
        transformed_df_row_count: int,
        additional_error_msg: str = ""):
    """Calculate number of dropped rows after transformations and throw error if too many rows lost."""
    if original_df_row_count <= 0:
        print("Original df contains no rows. Returning without calculating drop rate.")
        return
    drop_rate = (original_df_row_count - transformed_df_row_count) / original_df_row_count
    print(f"Calculated the df drop rate as {drop_rate*100}%. Comparing to threshold criteria...")

    if drop_rate > 0.10 and drop_rate <= 0.33:
        print(f"{drop_rate*100}% data missing after applying preprocessing."
              " Please check your log quality and the stdout logs for any possible debugging info."
              f" {additional_error_msg}")
    elif drop_rate > 0.33:
        raise InvalidInputError(
            f"Majority of the logs missing after applying preprocessing (drop_rate = {drop_rate*100}%)."
            " Failing preprocessing job. Please check your log quality and"
            " the stdout logs for any possible debugging info for help."
            f" {additional_error_msg}")
