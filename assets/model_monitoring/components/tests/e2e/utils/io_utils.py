# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains IO utility functions."""

import uuid
import pyspark.sql as pyspark_sql
from ruamel.yaml import YAML
from pyspark.sql import SparkSession


def load_from_yaml(path: str):
    """Load a yaml file into a dict."""
    with open(path, "r") as stream:
        return YAML().load(stream)


def write_to_yaml(path: str, contents: dict):
    """Write a dict to a yaml."""
    with open(path, "w+", encoding="utf8") as outfile:
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.dump(contents, outfile)


def print_header(msg: str):
    """Log a message header."""
    print("------------------------------")
    print(msg)
    print("------------------------------")


def generate_random_filename(extension: str):
    """Generate a GUID-based random file name with the given file extension."""
    return f"{str(uuid.uuid4())}.{extension}"


def create_pyspark_dataframe(data: list, columns: list) -> pyspark_sql.DataFrame:
    """Read mltable in spark."""
    spark = SparkSession.builder.getOrCreate()
    return spark.createDataFrame(data, columns)
