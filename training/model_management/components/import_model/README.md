## Import model

### Name 

import_model

### Version 

0.0.3

### Type 

pipeline

### Description 

Import a model into a workspace or a registry.
## Inputs 

pipeline specific compute

| Name    | Description                 | Type   | Default | Optional | Enum |
| ------- | --------------------------- | ------ | ------- | -------- | ---- |
| compute | Compute to run pipeline job | string |         |          |      |

Inputs for download model

| Name         | Description                                                                                                                                                                                                                                  | Type   | Default     | Optional | Enum                                |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ----------- | -------- | ----------------------------------- |
| model_source | Storage containers from where model will be sourced from                                                                                                                                                                                     | string | Huggingface |          | ['AzureBlob', 'GIT', 'Huggingface'] |
| model_id     | A valid model id for the model source selected. For example you can specify `bert-base-uncased` for importing HuggingFace bert base uncased model. Please specify the complete URL if **GIT** or **AzureBlob** is selected in `model_source` | string |             |          |                                     |

Inputs for the MlFLow conversion

| Name              | Description                                       | Type     | Default | Optional | Enum                                                                                                                                                                                                  |
| ----------------- | ------------------------------------------------- | -------- | ------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| license_file_path | Path to the license file                          | uri_file |         | True     |                                                                                                                                                                                                       |
| task_name         | A Hugging face task on which model was trained on | string   |         | True     | ['text-classification', 'fill-mask', 'token-classification', 'question-answering', 'summarization', 'text-generation', 'text-classification', 'translation', 'image-classification', 'text-to-image'] |

Inputs for Model registration

| Name              | Description                                                                                                                                         | Type     | Default | Optional | Enum |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------- | -------- | ---- |
| custom_model_name | Model name to use in the registration. If name already exists, the version will be auto incremented                                                 | string   |         | True     |      |
| model_version     | Model version in workspace/registry. If the same model name and version exists, the version will be auto incremented                                | string   |         | True     |      |
| model_description | Description of the model that will be shown in AzureML registry or workspace                                                                        | string   |         | True     |      |
| registry_name     | Name of the AzureML asset registry where the model will be registered. Model will be registered in a workspace if this is unspecified               | string   |         | True     |      |
| model_metadata    | A JSON or a YAML file that contains model metadata confirming to Model V2 [contract](https://azuremlschemas.azureedge.net/latest/model.schema.json) | uri_file |         | True     |      |

Pipeline outputs

| Name | Description | Type | Default | Optional | Enum |
| ---- | ----------- | ---- | ------- | -------- | ---- |

## Outputs 

| Name                       | Description                                                                                    | Type         |
| -------------------------- | ---------------------------------------------------------------------------------------------- | ------------ |
| mlflow_model_folder        | Output path for the converted MLFlow model                                                     | mlflow_model |
| model_registration_details | Output file which captures transformations applied above and registration details in JSON file | uri_file     |