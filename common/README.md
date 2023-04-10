# General Purpose Asset
## Overview
This directory contains the general purpose assets that includes [components](https://docs.microsoft.com/en-us/azure/machine-learning/concept-component) and [environments](https://docs.microsoft.com/en-us/azure/machine-learning/concept-environments).

## Structure of this folder

| Directory         | Description                                                                          |
|:------------------|:-------------------------------------------------------------------------------------|
| `components/deploy_model` | Definition for the component deploy_model.                                                      |
| `components/register_model`       | Definition for the component register_model.                                   |
| `src/deploy_model`  | Python code for deploy model component. |
| `src/register_model`          | Python code for register model component.                                                       |
| `environments`  | Definition for the environment python-sdk-v2  |

## Getting Started
1. Import the required library
```terminal
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
```

2. Get ML Client for `azureml` registry
```terminal
credential = DefaultAzureCredential()
ml_client_registry = MLClient(credential=credential, registry_name='azureml')
```

3. Import the components from azureml registry
```terminal
registration_component = ml_client_registry.components.get(name="register_model", version="0.0.1")
deployment_component = ml_client_registry.components.get(name="deploy_model", version="0.0.1")
```