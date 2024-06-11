# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Monitor Create Manifest Component."""

import argparse
import glob
import json
import os
import uuid
from shared_utilities.amlfs import amlfs_put_as_json, amlfs_download, amlfs_upload
from shared_utilities.io_utils import init_spark


def _generate_manifest(root_dir: str):

    manifest = {"version": "1.0.0", "metricsFiles": {}}

    for signal_files in glob.glob(os.path.join(root_dir, "signals/*.json")):
        with open(signal_files, "r") as fp:
            signal = json.loads(fp.read())
            manifest["metricsFiles"][
                signal["signalName"]
            ] = f"signals/{os.path.basename(signal_files)}"
    return manifest


def run():
    """Create Manifest."""
    # TODO: investigate why these aren't initialized by AzureML Filesystem itself
    # init env-vars for supporting credential-less datastore scenarios.
    # this env variable is required by Amlfs to get user token.
    spark = init_spark()
    spark_conf = spark.sparkContext.getConf()
    spark_conf_vars = {
        "AZUREML_SYNAPSE_CLUSTER_IDENTIFIER": "spark.synapse.clusteridentifier",
        "AZUREML_SYNAPSE_TOKEN_SERVICE_ENDPOINT": "spark.tokenServiceEndpoint",
    }
    for env_key, conf_key in spark_conf_vars.items():
        value = spark_conf.get(conf_key)
        if value:
            os.environ[env_key] = value

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_outputs_1", type=str, required=True)
    parser.add_argument("--model_monitor_metrics_output", type=str)

    for i in range(2, 10):
        parser.add_argument(
            f"--signal_outputs_{i}", type=str, required=False, nargs="?"
        )

    args = parser.parse_args()
    args_dict = vars(args)

    signals_outputs = []
    for i in range(1, 10):
        if args_dict[f"signal_outputs_{i}"] is None:
            continue
        print(str(args_dict[f"signal_outputs_{i}"]))
        signals_outputs.append(str(args_dict[f"signal_outputs_{i}"]))

    temp_path = str(uuid.uuid4())
    for signal_output in signals_outputs:
        amlfs_download(remote_path=signal_output, local_path=temp_path)
    amlfs_upload(local_path=temp_path, remote_path=args.model_monitor_metrics_output)
    amlfs_put_as_json(
        _generate_manifest(temp_path),
        args.model_monitor_metrics_output,
        "manifest.json",
    )

    print("*************** output metrics ***************")
    print("Successfully executed the create manifest component.")


if __name__ == "__main__":
    run()
