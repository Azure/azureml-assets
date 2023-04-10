# General Purpose Asset
## Overview
This directory contains the general purpose assets that includes [components](https://docs.microsoft.com/en-us/azure/machine-learning/concept-component) and [environments](https://docs.microsoft.com/en-us/azure/machine-learning/concept-environments).

## Structure of this folder

| Directory         | Description                                                                          |
|:------------------|:-------------------------------------------------------------------------------------|
| `components/deploy_model` | Definition of the component for the [model deployment](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=azure-cli).                                                      |
| `components/register_model`       | Definition for the component [model registration](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models?view=azureml-api-2&tabs=cli%2Cuse-local).                                   |
| `src/deploy_model`  | Python code for deploy model component. |
| `src/register_model`          | Python code for register model component.                                                       |
| `environments`  | Definition for the environment python-sdk-v2  |

## Examples
[Create pipeline using component assets registered in azureml](https://github.com/Azure/azureml-examples/blob/hrishikesh/workflow/sdk/python/foundation-models/system/import/import_model_into_registry.ipynb)
