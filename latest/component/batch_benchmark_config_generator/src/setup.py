# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AzureML Benchmarking SDK setup file."""

from setuptools import setup

setup(
    name="aml-benchmark",
    version="0.0.1",
    description="AzureML Benchmarking SDK",
    author="Microsoft Corp",
    entry_points={},
    install_requires=[
        "mlflow>=2.6.0",
        "azureml-mlflow>=1.52.0",
        "azure-ai-ml>=1.9.0",
        "azure-identity>=1.14.0",
        "azureml-core>=1.52.0",
        "azureml-telemetry>=1.52.0",
        "azureml-dataprep[pandas]>=4.12.0",
        "mltable>=1.5.0",
        "datasets"
    ],
    python_requires=">=3.8,<3.12",
)
