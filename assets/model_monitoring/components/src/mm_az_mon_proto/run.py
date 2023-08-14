# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for az mon metric Component."""

import argparse
import datetime
import os
import requests

from azure.ai.ml.identity import AzureMLOnBehalfOfCredential


def run():
    """Prototype to public az mon metrics."""
    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument("--baseline_histogram", type=str, required=False)

    args = arg_parser.parse_args()

    # print("====Job Args====")
    # for key, value in args.items():
    #     print(f'Job arg {key}: {value}')

    print("====Env Vars====")
    for key, value in os.environ.items():
        print(f'Env var {key}: {value}')

    # url = "https://eastus2euap.monitoring.azure.com/subscriptions/ea4faa5b-5e44-4236-91f6-5483d5b17d14/resourceGroups/model-monitoring-canary-rg/providers/Microsoft.MachineLearningServices/workspaces/model-monitoring-canary-ws/metrics"
    
    ws_resource_id = os.environ['AZUREML_WORKSPACE_SCOPE']
    uai_client_id = os.environ['AZUREML_IDENTITY_CLIENT_ID']
    
    credential = AzureMLOnBehalfOfCredential()
    access_token = credential.get_token('https://monitor.azure.com')
    
    print(f"access token: {access_token.token}")
    public_metrics(ws_resource_id, access_token.token)


def public_metrics(subject_resource_id: str, auth_token: str):
    payload = build_metric()
    headers = {
        "Authorization": f"Bearer {auth_token}"
    }

    x = requests.post(subject_resource_id, json = payload, headers=headers)

    print("Metric API response: " + x.text)


def build_metric():
    payload = {
        "time": datetime.datetime.utcnow().isoformat(" "),
        "data": {
            "baseData": {
                "metric": "az-mon-proto-signal",
                "namespace": "MyModelYeta",
                "dimNames": [
                    "Metrics",
                    "Date",
                ],
                "series": [
                    {
                        "dimValues": [
                            "JSD",
                            "810",
                        ],
                        "min": 1,
                        "max": 10,
                        "sum": 101,
                        "count": 1,
                    }
                ]
            },
        }
    }

    return payload

if __name__ == "__main__":
    run()
