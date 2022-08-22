### What are the assets being proposed?
**Component**
  Model Evaluation component

**Environment**
  Custom Environment for Model Evaluation
  Command component specific environment will be shared, which will contain following assets: 

  1. [Dockerfile](https://microsoft.sharepoint.com/:u:/t/SDAutoML/EbYHBgGmkmNNjUIvWEyci3gBT-Uqu73nwf5UPMj02BP0ow?e=G5wxcE)
  2. env.yaml

  
### Why should this asset be built-in?

Model Evaluation would be a part of DPV2 which allows user to run Model Evaluation for Any Machine Learning model on Azure ML.  In order for us to make the functionality of model evaluation available as generic functionality at Azure Machine Learning level, this component should be available as built-in componenet which users should be able to use it as drag and drop in designer.

Model Evaluation will be a standalone component available to user in all 3 user-experience. Currently, we have implemented the model evaluation feature as Command Component. SDK and CLI internally create a pipeline job which consumes model evaluation component. 3 user experiences as below: 

> V2 SDK : Model Evaluation Job can be created by user by consuming our component under a pipeline job. Our component can be a part of another pipeline or a standalone job as well. A sample code on how to consume our component using sdk is shown below:  

Sample Usage: [run_sdk.py](https://microsoft.sharepoint.com/:u:/t/SDAutoML/EZfUQhn23A9MssukOMQFFAEB_p47M2tuCiV83-9FtJPnng?e=ZNxexu)


  
> Azure ML CLI  : Similarly, Model evaluation job can also be created using Azure ML CLI. A User has to create a Pipeline job YAML with component as ‘azureml:model_evaluation:<version>’ and specify all other input parameters including Test data which is passed as URI_FOLDER.   
  
 Sample YAML: [model_evaluation_job.yml](https://microsoft.sharepoint.com/:u:/t/SDAutoML/EQWoVaptR05FiWmAHv3bCHIBHpLkEpGW8yz_wu2HV-nfJA?e=ecjFzG)  

  
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