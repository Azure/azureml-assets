## Convert model to MLFlow

### Name 

convert_model_to_mlflow

### Version 

0.0.3

### Type 

command

### Description 

Component converts the input model to MLFlow packaging format.

## Inputs 

| Name                    | Description                                                                                                                                                                    | Type       | Default | Optional | Enum                                                                                                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | ------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model_id                | Huggingface model id (https://huggingface.co/<model_id>). A required parameter for transformers flavor. Can be provided as input here or in model_download_metadata JSON file. | string     |         | True     |                                                                                                                                                                                                       |
| task_name               | A Hugging face task on which model was trained on. A required parameter for transformers mlflow flavor. Can be provided as input here or in model_download_metadata JSON file. | string     |         | True     | ['text-classification', 'fill-mask', 'token-classification', 'question-answering', 'summarization', 'text-generation', 'text-classification', 'translation', 'image-classification', 'text-to-image'] |
| model_download_metadata | JSON file containing model download details.                                                                                                                                   | uri_file   |         | True     |                                                                                                                                                                                                       |
| model_path              | Path to the model.                                                                                                                                                             | uri_folder |         | False    |                                                                                                                                                                                                       |
| license_file_path       | Path to the license file                                                                                                                                                       | uri_file   |         | True     |                                                                                                                                                                                                       |

## Outputs 

| Name                  | Description                                           | Type         |
| --------------------- | ----------------------------------------------------- | ------------ |
| mlflow_model_folder   | Output path for the converted MLFlow model.           | mlflow_model |
| model_import_job_path | JSON file containing model job path for model lineage | uri_file     |