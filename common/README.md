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

## Examples
[Create pipeline using component assets registered in azureml](https://github.com/Azure/azureml-examples/blob/hrishikesh/workflow/sdk/python/foundation-models/system/import/import_model_into_registry.ipynb)
