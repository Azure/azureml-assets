# General Purpose Assets
## Overview
This directory contains the general purpose assets that include [components](https://docs.microsoft.com/en-us/azure/machine-learning/concept-component) and [environments](https://docs.microsoft.com/en-us/azure/machine-learning/concept-environments) that can be used for purposes like [model registration](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models?view=azureml-api-2&tabs=cli%2Cuse-local) and [model deployment](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=azure-cli) steps in [Azure Machine Learning pipelines](https://learn.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines?view=azureml-api-2).

## Structure of this folder

| Directory | Description |
|:---|:---|
| `components/deploy_model` | Definition of the model deployment component. | 
| `components/register_model` | Definition for the model registration component. |
| `environments` | Definition for the python-sdk-v2 environment. |

## Examples
[Create pipeline using component assets registered in Azure ML](https://github.com/Azure/azureml-examples/blob/hrishikesh/workflow/sdk/python/foundation-models/system/import/import_model_into_registry.ipynb)

