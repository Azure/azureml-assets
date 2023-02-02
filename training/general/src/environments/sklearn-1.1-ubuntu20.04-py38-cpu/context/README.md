# SKLearn Docker Environment 

## Overview
This document provides details on how to create and run a sample workload in a containerized SKLearn Environment using stock version of SKLearn as well as Intel Extension for SKLearn on Azure ML instances.

## How it works
To create a stock SKLearn docker container use the `Dockerfile` and `conda_dependencies.yaml`. Follow the instructions below
```
sed -i 's#{{latest-image-tag}}#20230120.v1#g' Dockerfile 
docker build --build-arg LATEST_IMAGE_TAG=latest -f Dockerfile -t azure-ml:stock-sklearn .
```

To create SKLearn docker container using Intel Extensions,use `Dockerfile` and `conda_dependencies.yaml` inside `intel-extension` folder. Follow the instructions below
```
sed -i 's#{{latest-image-tag}}#20230120.v1#g' Dockerfile 
docker build --build-arg LATEST_IMAGE_TAG=latest -f Dockerfile -t azure-ml:intel-ex-sklearn .
```

## Run a Linear Regression workload inside the container
The code by default has Intel Extension Optimizations disabled.

This example shows how to use the `azure-ml:intel-ex-sklearn` container. Use the following commands:
```
WORKDIR=/workspace
PROJECT_DIR=<path/to/sklearn-1.1-ubuntu20.04-py38-cpu>
docker run -it -w $WORKDIR -v ${PROJECT_DIR}:${WORKDIR} azure-ml:intel-ex-sklearn /bin/bash 
```
Use the code available in `/workspace/tests/sklearn_sample_test.py` on an Azure ML instance. This code uses the `main.py` python script located in `/workspace/tests/src`. 

To enable Intel Extension for SKLearn, pass `command="python main.py --diabetes-csv ${{inputs.diabetes}} --intel-extension True"` to line 43 of `/workspace/tests/sklearn_sample_test.py`.

## Other Examples 
Refer to Code samples for various workloads available [here](https://github.com/intel/scikit-learn-intelex/tree/master/examples/notebooks#snake-intelr-extension-for-scikit-learn-notebooks) to leverage the benefits of Intel Extension for SKLearn.


