# General Purpose Assets
## Overview
This directory contains the general purpose assets that includes [components](https://docs.microsoft.com/en-us/azure/machine-learning/concept-component) and [environments](https://docs.microsoft.com/en-us/azure/machine-learning/concept-environments), which can be used for purpose like [model registration](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models?view=azureml-api-2&tabs=cli%2Cuse-local) and [model deployment](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=azure-cli) steps in [azure machine learning pipelines](https://learn.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines?view=azureml-api-2).

## Structure of this folder

| Directory         | Description                                                                          |
|:------------------|:-------------------------------------------------------------------------------------|
| `components/deploy_model` | Definition of the component for the model deployment.                                                      |
| `components/register_model`       | Definition for the component model registration.                                   |
| `environments`  | Definition for the environment python-sdk-v2  |

## Examples
[Create pipeline using component assets registered in azureml](https://github.com/Azure/azureml-examples/blob/hrishikesh/workflow/sdk/python/foundation-models/system/import/import_model_into_registry.ipynb)
