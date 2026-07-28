# OSS Distillation Batch Score Data Generation Pipeline

This component generates data from a teacher model endpoint by invoking it in batch mode. It is part of the OSS Distillation pipeline.

## Component Details

- **Name**: `oss_distillation_batchscoring_datagen_pipeline`
- **Version**: `0.0.1`
- **Type**: `pipeline`
- **Display Name**: `OSS Distillation Batch Score Data Generation Pipeline`
- **Description**: Component to generate data from teacher model endpoint by invoking it in batch.

## Inputs

| Name                             | Type     | Optional | Default                          | Description                                                                                                 |
|----------------------------------|----------|----------|----------------------------------|-------------------------------------------------------------------------------------------------------------|
| instance_type_pipeline_validation| string   | True     |                                  | Instance type to be used for validation component. The parameter compute_pipeline_validation must be set to 'serverless' for instance_type to be used. |
| instance_type_data_generation    | string   | True     | Standard_D4as_v4                 | Instance type to be used for finetune component in case of virtual cluster compute.                         |
| instance_type_data_import        | string   | True     | Singularity.ND96amrs_A100_v4     | Instance type to be used for data_import component in case of virtual cluster compute.                      |
| instance_type_finetune           | string   | True     | Singularity.ND96amrs_A100_v4     | Instance type to be used for finetune component in case of virtual cluster compute.                         |
| compute_pipeline_validation      | string   | True     | serverless                       | Compute to be used for validation component.                                                                |
| compute_data_generation          | string   | True     | serverless                       | Compute to be used for model_import.                                                                        |
| compute_data_import              | string   | True     | serverless                       | Compute to be used for model_import.                                                                        |
| compute_finetune                 | string   | True     | serverless                       | Compute to be used for finetune.                                                                            |
| train_file_path                  | uri_file | False    |                                  | Path to the registered training data asset.                                                                 |
| validation_file_path             | uri_file | True     |                                  | Path to the registered validation data asset.                                                               |
| teacher_model_endpoint_url       | string   | True     |                                  | Teacher model endpoint URL.                                                                                 |
| teacher_model_asset_id           | string   | True     |                                  | Teacher model Asset Id.                                                                                     |
| teacher_model_endpoint_name      | string   | True     |                                  | Teacher model endpoint name.                                                                                |
| teacher_model_max_new_tokens     | integer  | True     | 128                              | Teacher model max_new_tokens inference parameter.                                                           |
| teacher_model_temperature        | number   | True     | 0.2                              | Teacher model temperature inference parameter.                                                              |
| teacher_model_top_p              | number   | True     | 0.1                              | Teacher model top_p inference parameter.                                                                    |
| teacher_model_frequency_penalty  | number   | True     | 0.0                              | Teacher model frequency penalty inference parameter.                                                        |
| teacher_model_presence_penalty   | number   | True     | 0.0                              | Teacher model presence penalty inference parameter.                                                         |
| teacher_model_stop               | string   | True     |                                  | Teacher model stop inference parameter.                                                                     |
| min_endpoint_success_ratio       | number   | True     | 0.7                              | Minimum value of (successful_requests / total_requests) required for classifying inference as successful.   |
| enable_chain_of_thought          | string   | True     | false                            | Enable Chain of thought for data generation.                                                                |
| enable_chain_of_density          | string   | True     | false                            | Enable Chain of density for text summarization.                                                             |
| max_len_summary                  | integer  | True     | 80                               | Maximum Length Summary for text summarization.                                                              |
| data_generation_task_type        | string   | False    |                                  | Data generation task type. Supported values: NLI, CONVERSATION, NLU_QA, MATH, SUMMARIZATION.                |
| num_train_epochs                 | integer  | True     | 1                                | Training epochs.                                                                                            |
| per_device_train_batch_size      | integer  | True     | 1                                | Train batch size.                                                                                           |
| learning_rate                    | number   | True     | 3e-04                            | Start learning rate.                                                                                        |
| authentication_type              | string   | False    | azureml_workspace_connection     | Authentication type for endpoint. Supported values: azureml_workspace_connection, managed_identity.         |
| connection_name                  | string   | True     |                                  | Connection name to be used for authentication.                                                              |
| additional_headers               | string   | True     |                                  | JSON serialized string expressing additional headers to be added to each request.                           |
| debug_mode                       | boolean  | False    | False                            | Enable debug mode to print all the debug logs in the score step.                                            |
| ensure_ascii                     | boolean  | False    | False                            | If set to true, the output is guaranteed to have all incoming non-ASCII characters escaped.                 |
| max_retry_time_interval          | integer  | True     |                                  | The maximum time (in seconds) spent retrying a payload.                                                     |
| initial_worker_count             | integer  | False    | 5                                | The initial number of workers to use for scoring.                                                           |
| max_worker_count                 | integer  | False    | 200                              | Overrides `initial_worker_count` if necessary.                                                              |
| instance_count                   | integer  | False    | 1                                | Number of nodes in a compute cluster we will run the batch score step on.                                   |
| max_concurrency_per_instance     | integer  | False    | 1                                | Number of processes that will be run concurrently on any given node.                                        |
| mini_batch_size                  | string   | True     | 100KB                            | The mini batch size for parallel run.                                                                       |

## Outputs

| Name                             | Type     | Description                                                                                                 |
|----------------------------------|----------|-------------------------------------------------------------------------------------------------------------|
| generated_batch_train_file_path  | uri_file | Generated train data                                                                                        |
| generated_batch_validation_file_path | uri_file | Generated validation data                                                                                   |
