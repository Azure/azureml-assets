# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Contains functionality to patch pyspark classes for MLTable read and write support."""

def _patch_spark_dataframereader_format():
    from functools import update_wrapper
    from logging import warning
    from pyspark.sql import DataFrameReader, SparkSession, SQLContext
    from azureml.dataprep.api._spark_helper import read_spark_dataframe

    class MLTableReader(object):
        def __init__(self, dataframe_reader):
            self._dataframe_reader = dataframe_reader
            self.__copy_members()

        def __copy_members(self):
            for member_name in dir(self._dataframe_reader):
                member = getattr(self._dataframe_reader, member_name)
                if not hasattr(self, member_name):
                    setattr(self, member_name, member)

        def load(self, uri, storage_options: dict = None):
            import mltable
            import os
            warning('spark.read.mltable ignores any options set on the DataFrameReader'
                    ' and only uses parsing options specified in the MLTable file.')

            if isinstance(self._dataframe_reader._spark, SparkSession):
                spark_session = self._dataframe_reader._spark
            elif isinstance(self._dataframe_reader._spark, SQLContext):
                spark_session = self._dataframe_reader._spark.sparkSession
            else:
                raise(f'Unsupported type for spark context: {type(self._dataframe_reader._spark)}')

            spark_conf = spark_session.sparkContext.getConf()
            spark_conf_vars = {
                'AZUREML_SYNAPSE_CLUSTER_IDENTIFIER': 'spark.synapse.clusteridentifier',
                'AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT': 'spark.tokenServiceEndpoint',
            }

            aux_envvars = {
                'AZUREML_RUN_TOKEN': os.environ['AZUREML_RUN_TOKEN'],
            }

            for env_key, conf_key in spark_conf_vars.items():
                value = spark_conf.get(conf_key)
                if value:
                    aux_envvars[env_key] = value
                    os.environ[env_key] = value

            table = mltable.load(uri, storage_options)
            # TODO: remove the fallback invocation once dataprep and mltable have released
            try:
                return read_spark_dataframe(table._dataflow.to_yaml_string(), spark_session, aux_envvars)
            except TypeError:
                return read_spark_dataframe(table._dataflow.to_yaml_string(), spark_session)

    original_method = DataFrameReader.format

    def format(dataframe_reader, source):
        if source == 'mltable':
            return MLTableReader(dataframe_reader)
        else:
            return original_method(dataframe_reader, source)

    DataFrameReader.format = update_wrapper(format, original_method)


def _patch_spark_dataframereader_mltable():
    from pyspark.sql import DataFrameReader

    def mltable(dataframe_reader, uri, storage_options: dict = None):
        return dataframe_reader.format('mltable').load(uri, storage_options)

    DataFrameReader.mltable = mltable


class _DataframeWriterManager(object):
    def __init__(self, dataframe, uri, **kwargs):
        self.dataframe = dataframe
        self.uri = uri
        self.total_partitions = dataframe.rdd.getNumPartitions()
        self.output_format = kwargs.get("output_format", "delimited")
        self.delimiter = kwargs.get("delimiter", ",")
        self.overwrite = kwargs.get("overwrite", False)
        self.encoding = kwargs.get("encoding", "utf8")

        if self.output_format == "parquet":
            self.extension = "parquet"
        elif self.output_format == "delimited":
            self.extension = "csv"
        elif self.output_format == "json_lines":
            self.extension = "json"
        else:
            raise NotImplementedError(
                f'Unsupported output format "{self.output_format}".'
                + ' Supported formats are "parquet", "delimited" and "json_lines".'
            )

    def write(self):
        mode = "overwrite" if self.overwrite else "errorifexists"

        if self.output_format == "parquet":
            self.dataframe.write.mode(mode).parquet(self.uri)
        elif self.output_format == "delimited":
            self.dataframe.write.mode(mode).option("delimiter", self.delimiter).option(
                "header", True
            ).option("encoding", self.encoding).csv(self.uri)
        elif self.output_format == "json_lines":
            self.dataframe.write.mode(mode).option("encoding", self.encoding).json(
                self.uri
            )
        else:
            raise NotImplementedError(
                f'Unsupported output format "{self.output_format}".'
                + 'Supported formats are "parquet", "delimited" and "json_lines".'
            )

        base_path = self.uri.rstrip("/")
        return base_path + "/*." + self.extension


def _write_mltable_yaml(uri, output_path_pattern, manager):
    from azureml.dataprep.fuse.dprepfuse import MountOptions, rslex_uri_volume_mount
    import os
    import tempfile
    import yaml

    mltable_obj = {
        "paths": [{"pattern": output_path_pattern}],
    }

    if manager.output_format == "parquet":
        mltable_obj["transformations"] = ["read_parquet"]
    elif manager.output_format == "delimited":
        mltable_obj["transformations"] = [
            {
                "read_delimited": {
                    "delimiter": manager.delimiter,
                    "encoding": manager.encoding,
                    "header": "all_files_same_headers",
                }
            },
        ]
    elif manager.output_format == "json_lines":
        mltable_obj["transformations"] = [
            {
                "read_json_lines": {
                    "encoding": manager.encoding,
                    "invalid_lines": "error",
                    "include_path_column": False,
                }
            },
        ]

    with tempfile.TemporaryDirectory() as temp_dir:
        with rslex_uri_volume_mount(
            uri=uri, mount_point=temp_dir, options=MountOptions(read_only=False)
        ):
            with open(os.path.join(temp_dir, "MLTable"), "w") as yaml_file:
                yaml.dump(mltable_obj, yaml_file)


def _write_spark_dataframe(dataframe, uri, **kwargs):
    from shared_utilities.io_utils import convert_to_azureml_uri
    manager = _DataframeWriterManager(dataframe, uri, **kwargs)

    output_path_pattern = manager.write()
    output_path_pattern_azureml = convert_to_azureml_uri(output_path_pattern)

    _write_mltable_yaml(uri, output_path_pattern_azureml, manager)


def _patch_spark_dataframewriter_format():
    from functools import update_wrapper
    from pyspark.sql import DataFrameWriter

    class MLTableWriter(object):
        def __init__(self, dataframe_writer):
            self._dataframe_writer = dataframe_writer
            self.__copy_members()

        def __copy_members(self):
            for member_name in dir(self._dataframe_writer):
                member = getattr(self._dataframe_writer, member_name)
                if not hasattr(self, member_name):
                    setattr(self, member_name, member)

        def save(self, uri, **kwargs):
            """
            Save the content of the :class:`DataFrame` as an MLTable artifact at the specified path.

            Parameters
            ----------
            uri : str
                string represents path to the MLTable directory, and supports long-form
                datastore uri, storage uri or local path.

            Other Parameters
            ----------------
            output_format: str
                string indicating the format to emit data in. Accepted values are 'parquet', 'delimited',
                and 'json_lines'. Defaults to 'delimited'.
            delimiter: str
                the delimiter to use when emitting data. Defaults to ','.
                Ignored unless output_format is 'delimited'.
            overwrite: bool
                boolean indicating whether to overwrite files if they already exist.
                Throws if one or more files is present, and overwrite is set to False. Defaults to False.
            encoding: str
                string representing the encoding format to use when emitting data.
                Ignored unless output_format is 'delimited' or 'json_lines'. Defaults to 'utf8'.

            Examples
            --------
            >>> df.write.format('mltable').save('azureml://subscriptions/<subscription-id>/
                        resourcegroups/<resourcegroup-name>/workspaces/<workspace-name>/
                        datastores/<datastore-name>/paths/<mltable-path-on-datastore>/')
            """
            if hasattr(self, "_optionsdict"):
                options = {**self._optionsdict, **kwargs}
            else:
                options = kwargs

            _write_spark_dataframe(self._dataframe_writer._df, uri, **options)

    original_method = DataFrameWriter.format

    def format(dataframe_writer, source):
        if source == "mltable":
            return MLTableWriter(dataframe_writer)
        else:
            return original_method(dataframe_writer, source)

    DataFrameWriter.format = update_wrapper(format, original_method)


def _patch_spark_dataframewriter_option():
    from functools import update_wrapper
    from pyspark.sql import DataFrameWriter

    original_method = DataFrameWriter.option

    def option(dataframe_writer, key: str, value) -> "DataFrameWriter":
        if not hasattr(dataframe_writer, "_optionsdict"):
            dataframe_writer._optionsdict = dict()

        dataframe_writer._optionsdict[key] = value
        return original_method(dataframe_writer, key, value)

    DataFrameWriter.option = update_wrapper(option, original_method)


def _patch_spark_dataframewriter_options():
    from functools import update_wrapper
    from pyspark.sql import DataFrameWriter

    original_method = DataFrameWriter.options

    def options(dataframe_writer, **options) -> "DataFrameWriter":
        if not hasattr(dataframe_writer, "_optionsdict"):
            dataframe_writer._optionsdict = dict()

        dataframe_writer._optionsdict.update(options)
        return original_method(dataframe_writer, **options)

    DataFrameWriter.options = update_wrapper(options, original_method)


def _patch_spark_dataframewriter_mltable():
    from pyspark.sql import DataFrameWriter

    def mltable(dataframe_writer, uri, **kwargs):
        """
        Save the content of the :class:`DataFrame` as an MLTable artifact at the specified path.

        Parameters
        ----------
        uri : str
            string represents path to the MLTable directory, and supports long-form
            datastore uri, storage uri or local path.

        Other Parameters
        ----------------
        output_format: str
            string indicating the format to emit data in. Accepted values are 'parquet', 'delimited',
            and 'json_lines'. Defaults to 'delimited'.
        delimiter: str
            the delimiter to use when emitting data. Defaults to ','.
            Ignored unless output_format is 'delimited'.
        overwrite: bool
            boolean indicating whether to overwrite files if they already exist.
            Throws if one or more files is present, and overwrite is set to False. Defaults to False.
        encoding: str
            string representing the encoding format to use when emitting data.
            Ignored unless output_format is 'delimited' or 'json_lines'. Defaults to 'utf8'.

        Examples
        --------
        >>> df.write.mltable('azureml://subscriptions/<subscription-id>/
                    resourcegroups/<resourcegroup-name>/workspaces/<workspace-name>/
                    datastores/<datastore-name>/paths/<mltable-path-on-datastore>/')
        """
        if hasattr(dataframe_writer, "_optionsdict"):
            options = {**dataframe_writer._optionsdict, **kwargs}
        else:
            options = kwargs

        _write_spark_dataframe(dataframe_writer._df, uri, **options)

    DataFrameWriter.mltable = mltable


def patch_all():
    """Patch all."""
    _patch_spark_dataframereader_format()
    _patch_spark_dataframereader_mltable()

    _patch_spark_dataframewriter_format()
    _patch_spark_dataframewriter_mltable()
    _patch_spark_dataframewriter_option()
    _patch_spark_dataframewriter_options()

    import azureml.dataprep.api._spark_helper

    def _my_get_partitions(dataflow):
        return azureml.dataprep.api._dataframereader._execute(
            "_spark_helper._get_partitions",
            dataflow,
            collect_results=True,
            allow_fallback_to_clex=False,
        )[1]

    azureml.dataprep.api._spark_helper._get_partitions = _my_get_partitions
