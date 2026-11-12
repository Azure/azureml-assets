# SKLearn Docker Environment 

## Overview
This document provides details on how to create and run a sample workload in a containerized SKLearn Environment using stock version of SKLearn as well as Intel Extension for SKLearn on Azure ML instances.

## How it works
The container has stock and intel-optimized versions available. 
It uses Python 3.11 on the MCR Ubuntu 24.04/OpenMPI 5.0 base.
Intel Extension is pinned to `2025.0.1` to retain Scikit-learn 1.5 compatibility.
Newer Intel releases require Scikit-learn >=1.6 and have removed the older API fallbacks
([upstream change](https://github.com/uxlfoundation/scikit-learn-intelex/pull/3259)).
Revisit the pin only with an Intel release validated against Scikit-learn 1.5.
To build the docker container use the `Dockerfile` and `conda_dependencies.yaml`.
Refer to the link [here](https://github.com/Azure/azureml-assets/wiki/Environments#testing-environment-image-builds) on how the macros are replaced.

## Run a Linear Regression workload inside the container
The code by default has Intel Extension Optimizations disabled.

This example shows how to use the container. Use the following commands:
```
WORKDIR=/workspace
PROJECT_DIR=<path/to/sklearn-1.5>
IMAGE_NAME=<name of built image>
docker run -it -w $WORKDIR -v ${PROJECT_DIR}:${WORKDIR} $IMAGE_NAME /bin/bash 
```
Use the code available in `/workspace/tests/sklearn_sample_test.py` on an Azure ML instance inside the container. This code uses the `main.py` python script located in `/workspace/tests/src`. 

To enable Intel Extension for SKLearn, pass `command="python main.py --diabetes-csv ${{inputs.diabetes}} --intel-extension True"`.
The image test runs both modes and verifies that Intel mode uses an accelerated estimator.

## Other Examples 
Refer to Code samples for various workloads available [here](https://github.com/intel/scikit-learn-intelex/tree/master/examples/notebooks#snake-intelr-extension-for-scikit-learn-notebooks) to leverage the benefits of Intel Extension for SKLearn.

