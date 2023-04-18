## Register model

### Name 

register_model

### Version 

0.0.3

### Type 

command

### Description 

Register a model to a workspace or a registry. The component works on compute with [MSI](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-manage-compute-instance?tabs=python) attached. See [docs](https://aka.ms/azureml/components/register_model) to learn more.

## Inputs 

| Name                    | Description                                                                                                                                  | Type       | Default      | Optional | Enum                             |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------ | -------- | -------------------------------- |
| model_name              | Model name to use in the registration. If name already exists, the version will be auto incremented                                          | string     |              | True     |                                  |
| model_version           | Model version in workspace/registry. If the same model name and version exists, the version will be auto incremented                         | string     |              | True     |                                  |
| model_type              | Model type                                                                                                                                   | string     | mlflow_model | True     | ['custom_model', 'mlflow_model'] |
| model_description       | Description of the model that will be shown in AzureML registry or workspace                                                                 | string     |              | True     |                                  |
| registry_name           | Name of the AzureML asset registry where the model will be registered. Model will be registered in a workspace if this is unspecified        | string     |              | True     |                                  |
| model_path              | Path to the model directory                                                                                                                  | uri_folder |              | False    |                                  |
| model_download_metadata | A JSON file which contains information related to model download.                                                                            | uri_file   |              | True     |                                  |
| model_metadata          | JSON/YAML file that contains model metadata confirming to Model V2 [contract](https://azuremlschemas.azureedge.net/latest/model.schema.json) | uri_file   |              | True     |                                  |
| model_import_job_path   | JSON file that contains the job path of model to have lineage.                                                                               | uri_file   |              | True     |                                  |

## Outputs 

| Name                 | Description                                                     | Type     |
| -------------------- | --------------------------------------------------------------- | -------- |
| registration_details | JSON file into which model registration details will be written | uri_file |