# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Feature Retrieval Component."""

import argparse
import logging
import os

from azureml.dataprep.api._rslex_executor import ensure_rslex_environment
from azureml.dataprep.rslex import Copier, PyIfDestinationExists, PyLocationInfo
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azureml.featurestore import FeatureStoreClient, get_offline_features
from azureml.featurestore._utils._constants import FEATURE_RETRIEVAL_SPEC_YAML_FILENAME
from azureml.featurestore._utils.utils import _ensure_azureml_full_path

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

parser = argparse.ArgumentParser()
parser.add_argument("--observation_data", type=str, help="hdfs path of observation data")
parser.add_argument("--timestamp_column", type=str, help="entity df time series column name")
parser.add_argument("--input_model", required=False, type=str, help="model asset using features")
parser.add_argument("--feature_retrieval_spec", required=False, type=str, help="feature retrieval spec file")
parser.add_argument("--observation_data_format", type=str, help="observation data format")
parser.add_argument("--output_data", type=str, help="output path")
args, _ = parser.parse_known_args()

assert args.feature_retrieval_spec is not None or args.input_model is not None

if args.feature_retrieval_spec is not None and args.input_model is not None:
    raise Exception("Only one of input_model or feature_retrieval_spec should be provided.")

fs_client = FeatureStoreClient(credential=AzureMLOnBehalfOfCredential())

ensure_rslex_environment()

sub_id = os.environ["AZUREML_ARM_SUBSCRIPTION"]
rg = os.environ["AZUREML_ARM_RESOURCEGROUP"]
ws = os.environ["AZUREML_ARM_WORKSPACE_NAME"]

if args.input_model:
    feature_retrieval_spec_folder = args.input_model
else:
    feature_retrieval_spec_folder = args.feature_retrieval_spec

features = fs_client.resolve_feature_retrieval_spec(feature_retrieval_spec_folder)

entity_df_path = args.observation_data
if args.observation_data_format == "parquet":
    entity_df = spark.read.parquet(entity_df_path)
elif args.observation_data_format == "csv":
    entity_df = spark.read.csv(entity_df_path, header=True)
elif args.observation_data_format == "delta":
    entity_df = spark.read.format("delta").load(entity_df_path)
else:
    raise Exception("Please provide a valid observation_data_format type. " +
                    "Supported values are 'parquet', 'csv' and 'delta' ")

training_df = get_offline_features(
    features=features,
    observation_data=entity_df,
    timestamp_column=args.timestamp_column)

logger.info("Printing head of the generated data.")
logger.info(training_df.head(5))

logger.info("Outputting dataset to parquet files.")
training_df.write.mode("overwrite").parquet(os.path.join(args.output_data, "data/"))

# Write feature_retrieval_spec.yaml to the output_folder
if_destination_exists = PyIfDestinationExists.MERGE_WITH_OVERWRITE
dest_uri = PyLocationInfo.from_uri(_ensure_azureml_full_path(args.output_data, sub_id, rg, ws))
src_uri = os.path.join(feature_retrieval_spec_folder, FEATURE_RETRIEVAL_SPEC_YAML_FILENAME)

Copier.copy_uri(dest_uri, src_uri, if_destination_exists, "")

logger.info("Done!")
