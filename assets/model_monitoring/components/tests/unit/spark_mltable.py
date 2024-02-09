# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""mltable support for pyspark."""

import datetime


SPARK_ZIP_PATH = 'SPARK_ZIP_PATH'


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
                raise f'Unsupported type for spark context: {type(self._dataframe_reader._spark)}'

            sc = spark_session.sparkContext
            # if SPARK_ZIP_PATH is set, add the zip file to the spark context
            zip_path = os.environ.get(SPARK_ZIP_PATH, '')
            if zip_path:
                sc.addPyFile(zip_path)

            spark_conf = spark_session.sparkContext.getConf()
            spark_conf_vars = {
                'AZUREML_SYNAPSE_CLUSTER_IDENTIFIER': 'spark.synapse.clusteridentifier',
                'AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT': 'spark.tokenServiceEndpoint',
            }

            aux_envvars = {
                'AZUREML_RUN_TOKEN': os.environ.get('AZUREML_RUN_TOKEN', ''),
                'AZUREML_RUN_TOKEN_EXPIRY': os.environ.get('AZUREML_RUN_TOKEN_EXPIRY', ''),
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
        self.output_format = kwargs.get('output_format', 'delimited')
        self.delimiter = kwargs.get('delimiter', ',')
        self.overwrite = kwargs.get('overwrite', False)
        self.encoding = kwargs.get('encoding', 'utf8')

        if self.output_format == 'parquet':
            self.extension = 'parquet'
        elif self.output_format == 'delimited':
            self.extension = 'csv'
        elif self.output_format == 'json_lines':
            self.extension = 'json'
        else:
            raise NotImplementedError(f'Unsupported output format "{self.output_format}". \
                                      Supported formats are "parquet", "delimited" and "json_lines".')

    def write(self):
        mode = "overwrite" if self.overwrite else "errorifexists"

        if self.output_format == 'parquet':
            self.dataframe.write.mode(mode).parquet(self.uri)
        elif self.output_format == 'delimited':
            self.dataframe.write.mode(mode).option('delimiter', self.delimiter).option("header", True) \
                .option('encoding', self.encoding).csv(self.uri)
        elif self.output_format == 'json_lines':
            self.dataframe.write.mode(mode).option('encoding', self.encoding).json(self.uri)
        else:
            raise NotImplementedError(f'Unsupported output format "{self.output_format}". \
                                      Supported formats are "parquet", "delimited" and "json_lines".')

        base_path = self.uri.rstrip('/')
        return base_path + "/*." + self.extension


def _write_mltable_yaml(uri, output_path_pattern, manager):
    from azureml.dataprep.fuse.dprepfuse import MountOptions, rslex_uri_volume_mount
    import os
    import tempfile
    import yaml

    mltable_obj = {
        'paths': [{'pattern': output_path_pattern}],
    }

    if manager.output_format == 'parquet':
        mltable_obj['transformations'] = ['read_parquet']
    elif manager.output_format == 'delimited':
        mltable_obj['transformations'] = [{
            'read_delimited': {
                'delimiter': manager.delimiter,
                'encoding': manager.encoding,
                'header': 'all_files_same_headers'
            }
        }]
    elif manager.output_format == 'json_lines':
        mltable_obj['transformations'] = [{
            'read_json_lines': {
                'encoding': manager.encoding,
                'invalid_lines': 'error',
                'include_path_column': False
            }
        }]

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with rslex_uri_volume_mount(uri=uri, mount_point=temp_dir, options=MountOptions(read_only=False)):
                with open(os.path.join(temp_dir, 'MLTable'), 'w') as yaml_file:
                    yaml.dump(mltable_obj, yaml_file)
        except Exception as e:
            # if on a windows machine, return since azureml-dataprep-rslex does not support PyMountOptions
            if os.name == 'nt':
                return
            raise e


def _write_spark_dataframe(dataframe, uri, **kwargs):
    import os
    from pyspark.sql import SparkSession, SQLContext

    if isinstance(dataframe.write._spark, SparkSession):
        spark_session = dataframe.write._spark
    elif isinstance(dataframe.write._spark, SQLContext):
        spark_session = dataframe.write._spark.sparkSession
    else:
        raise f'Unsupported type for spark context: {type(dataframe.write._spark)}'

    spark_conf = spark_session.sparkContext.getConf()
    spark_conf_vars = {
        'AZUREML_SYNAPSE_CLUSTER_IDENTIFIER': 'spark.synapse.clusteridentifier',
        'AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT': 'spark.tokenServiceEndpoint',
    }

    aux_envvars = {
        'AZUREML_RUN_TOKEN': os.environ.get('AZUREML_RUN_TOKEN', ''),
        'AZUREML_RUN_TOKEN_EXPIRY': os.environ.get('AZUREML_RUN_TOKEN_EXPIRY', ''),
    }

    for env_key, conf_key in spark_conf_vars.items():
        value = spark_conf.get(conf_key)
        if value:
            aux_envvars[env_key] = value
            os.environ[env_key] = value

    manager = _DataframeWriterManager(dataframe, uri, **kwargs)

    output_path_pattern = manager.write()

    _write_mltable_yaml(uri, output_path_pattern, manager)


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
            if hasattr(self, '_optionsdict'):
                options = {**self._optionsdict, **kwargs}
            else:
                options = kwargs

            _write_spark_dataframe(self._dataframe_writer._df, uri, **options)

    original_method = DataFrameWriter.format

    def format(dataframe_writer, source):
        if source == 'mltable':
            return MLTableWriter(dataframe_writer)
        else:
            return original_method(dataframe_writer, source)

    DataFrameWriter.format = update_wrapper(format, original_method)


def _patch_spark_dataframewriter_option():
    from functools import update_wrapper
    from pyspark.sql import DataFrameWriter

    original_method = DataFrameWriter.option

    def option(dataframe_writer, key: str, value) -> DataFrameWriter:
        if not hasattr(dataframe_writer, '_optionsdict'):
            dataframe_writer._optionsdict = dict()

        dataframe_writer._optionsdict[key] = value
        return original_method(dataframe_writer, key, value)

    DataFrameWriter.option = update_wrapper(option, original_method)


def _patch_spark_dataframewriter_options():
    from functools import update_wrapper
    from pyspark.sql import DataFrameWriter

    original_method = DataFrameWriter.options

    def options(dataframe_writer, **options) -> DataFrameWriter:
        if not hasattr(dataframe_writer, '_optionsdict'):
            dataframe_writer._optionsdict = dict()

        dataframe_writer._optionsdict.update(options)
        return original_method(dataframe_writer, **options)

    DataFrameWriter.options = update_wrapper(options, original_method)


def _patch_spark_dataframewriter_mltable():
    from pyspark.sql import DataFrameWriter

    def mltable(dataframe_writer, uri, **kwargs):
        if hasattr(dataframe_writer, '_optionsdict'):
            options = {**dataframe_writer._optionsdict, **kwargs}
        else:
            options = kwargs

        _write_spark_dataframe(dataframe_writer._df, uri, **options)

    DataFrameWriter.mltable = mltable


def patch_all():
    """Patch all spark classes for MLTable support."""
    print("[" + str(datetime.datetime.now()) + "]" + "Start to import mltable.")
    try:
        import mltable
        print(mltable.__name__)
    except Exception as e:
        print(f"[{str(datetime.datetime.now())}]Skip patching mltable feature because import has exception: {e} ")
        return
    print(f"[{str(datetime.datetime.now())}]Start to patch spark_dataframereader_format.")
    _patch_spark_dataframereader_format()
    print(f"[{str(datetime.datetime.now())}]Start to patch spark_dataframereader_mltable.")
    _patch_spark_dataframereader_mltable()
    print(f"[{str(datetime.datetime.now())}]Start to patch spark_dataframewriter_format.")
    _patch_spark_dataframewriter_format()
    print(f"[{str(datetime.datetime.now())}]Start to patch spark_dataframewriter_mltable.")
    _patch_spark_dataframewriter_mltable()
    print(f"[{str(datetime.datetime.now())}]Start to patch spark_dataframewriter_option.")
    _patch_spark_dataframewriter_option()
    print(f"[{str(datetime.datetime.now())}]Start to patch spark_dataframewriter_options.")
    _patch_spark_dataframewriter_options()


try:
    patch_all()
except Exception as e:
    print("[" + str(datetime.datetime.now()) + "]" + f'Failed to patch pyspark classes for MLTable support: {e}')
print("[" + str(datetime.datetime.now()) + "]" + "Done patch.")
