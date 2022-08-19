### What are the assets being proposed?
**Component**
  Model Evaluation component

**Environment**
  Custom Environment for Model Evaluation
  Command component specific environment will be shared, which will contain following assets: 

  1. Dockerfile
  2. env.yaml

  
### Why should this asset be built-in?

Model Evaluation would be a part of DPV2 which allows user to run Model Evaluation for Any Machine Learning model on Azure ML.  In order for us to make the functionality of model evaluation available as generic functionality at Azure Machine Learning level, this component should be available as built-in componenet which users should be able to use it as drag and drop in designer.

Model Evaluation will be a standalone component available to user in all 3 user-experience. Currently, we have implemented the model evaluation feature as Command Component. SDK and CLI internally create a pipeline job which consumes model evaluation component. 3 user experiences as below: 

> V2 SDK : Model Evaluation Job can be created by user by consuming our component under a pipeline job. Our component can be a part of another pipeline or a standalone job as well. A sample code on how to consume our component using sdk is shown below:  

Sample Usage: run_sdk.py 

'''
"""Skeleton script for creating Model Evaluation Job"""
from azure.ai.ml.entities import PipelineJob
from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential
# Using DefaultAzureCredential()
credential = DefaultAzureCredential()

ml_client = None
try:
    ml_client = MLClient.from_config(credential)
except Exception as ex:
    # Enter details of your AML workspace
    subscription_id = "72c03bf3-4e69-41af-9532-dfcdc3eefef4"
    resource_group = "shared-model-evaluation-rg"
    workspace = "aml-shared-model-evaluation-ws"
    ml_client = MLClient(credential, subscription_id, resource_group, workspace)

def create_pipeline_job(
        data_folder,                           #Path to data folder
        test_data_file_name,
        mode,
        task,
        model_uri=None,                           
        mlflow_model=None,                     #Path to Mlflow model
        additional_parameters=None
    ):
    model_evaluation_component =  ml_client.components.get(name="model_evaluation", version="15")
    data_folder = Input(type="uri_folder", path=data_folder)
    if mlflow_model:
        mlflow_model = Input(type="mlflow_model", path=mlflow_model)
    else:
        if not model_uri:
            raise ValueError("Either model_uri or mlflow model should be passed")
    
    node = model_evaluation_component(
        data_folder=data_folder,
        mode=mode,
        task=task,
        test_data_file_name=test_data_file_name,
        mlflow_model=mlflow_model,
        model_uri=model_uri,
        additional_parameters=additional_parameters
    )
    pipeline_job = PipelineJob(jobs={"model_evaluation_job":node})
    pipeline_job.settings.default_compute = "model-eval-cpu-cluster"

job = create_pipeline_job(<args>)
submitted_job = ml_client.create_or_update(job, experiment_name="Model Evaluation Experiment")

print(submitted_job)
'''
  
> Azure ML CLI  : Similarly, Model evaluation job can also be created using Azure ML CLI. A User has to create a Pipeline job YAML with component as ‘azureml:model_evaluation:<version>’ and specify all other input parameters including Test data which is passed as URI_FOLDER.   
  
 Sample YAML: model_evaluation_job.yml  

'''
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
type: pipeline
display_name: Model Evaluation CLI Test
description: Model evaluation test using CLI v2
  
compute: cpu-cluster1

jobs:
  model_evaluation_job:
    type: command
    component: azureml:model_evaluation@latest
    inputs:
      mode: score
      task: classification
      model_uri: runs:/057b5838-f356-4bfa-a2e6-dd5ea479c1c2/model/
      data_folder:
        path: ./test_data/
        type: uri_folder
      test_data_file_name: bank_marketing_test_data.csv
      additional_parameters: metrics_config.json
'''
  
> Designer (UI)  : The model evaluation component will be available in Designer which allows user to drag and drop the component along with Test Data and filling in the rest of the component parameters and create a model evaluation job using UI.

Parameters:
|       Name                 |        Type     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|----------------------------|:---------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     mode                   |   String        |   Options: [score, predict, compute_metrics]  Required: True  Description: Model Evaluation mode ->         1. score -> Do prediction and compute metrics for predictions        2. predict -> Compute predictions only over given dataset        3. compute_metrics -> Compute metrics over predictions provided by user                                                                                                                                                                         |
|     task                   |   String        |   Options: [classification, regression, forecasting]  Required: True  Description: Task type for which the model is trained                                                                                                                                                                                                                                                                                                                                                                       |
|     model_uri              |   String        |   Required: False (Required only if mode is score or predict)  Description: MLFlow model uri (could be of type) ->        To Fetch models from an azureml run        - runs:/<azureml_run_id>/run-relative/path/to/model        To Fetch model from azureml model registry        - models:/<model_name>/<model_version>        - models:/<model_name>/<stage>  NOTE: model_uri has been added to accommodate current ML Designer as it doesn’t allow users to drag-and-drop Registered models    |
|     mlflow_model           |   MLFlow Model  |   Required: False (Required only if mode is score or predict)  Description: MLFlow model (either registered or output of another job)                                                                                                                                                                                                                                                                                                                                                             |
|     data_folder            |   URI Folder    |   Required: True  Description: Input folder which contains the following files ->        - Test Data (in csv format)        - Additional Parameters file (in JSON format)        - Any other artifact required for model evaluation like y_transformer (in pickled format)                                                                                                                                                                                                                        |
|     test_data_file_name    |   String        |   Required: True  Description: Full path of test data file in above folder                                                                                                                                                                                                                                                                                                                                                                                                                        |
|     additional_parameters  |   String        |   Required: False  Description: File name of the additional parameters JSON file.        Contents of JSON File:        - label_column_name: <Name of target column in test csv>                      (required if mode is score or compute_metrics)        - prediction_column_name: <Name of predictions column in test csv> (required if mode is compute_metrics)        - <Other parameters based on above task specific Metrics class from above SDK package>                                 |

### Support model (what teams will be on the hook for bug fixes and security patches)?
PM (Sharmeelee Bijlani)
Dev Lead (Shipra Jain, Anup Shirgaonkar)

### A high-level description of the implementation for each asset.

Model Evaluation would be a part of DPV2 which allows user to run Model Evaluation for Any Machine Learning model on Azure ML.  In order for us to make the functionality of model evaluation available as generic functionality at Azure Machine Learning level, this component should be available as built-in component which users should be able to use as drag and drop in designer.