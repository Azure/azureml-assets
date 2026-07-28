# OSS Distillation Generate Data Batch Scoring Preprocess

## Description
This component prepares data to invoke the teacher model endpoint in batch. It supports various data formats such as `jsonl`, `json`, `csv`, `tsv`, and `parquet`.

## Environment
The component uses the following environment:
- `azureml://registries/azureml/environments/acft-hf-nlp-gpu/labels/latest`

## Inputs
The component accepts the following inputs:

| Name                          | Type      | Description                                                                                     | Required | Default |
|-------------------------------|-----------|-------------------------------------------------------------------------------------------------|----------|---------|
| `train_file_path`             | uri_file  | Path to the registered training data asset. Supported formats: `jsonl`, `json`, `csv`, `tsv`, `parquet`. | Yes      |         |
| `validation_file_path`        | uri_file  | Path to the registered validation data asset. Supported formats: `jsonl`, `json`, `csv`, `tsv`, `parquet`. | No       |         |
| `teacher_model_endpoint_url`  | string    | The URL of the teacher model endpoint.                                                          | Yes      |         |
| `teacher_model_asset_id`      | string    | The asset ID of the teacher model.                                                              | Yes      |         |
| `teacher_model_max_new_tokens`| integer   | Teacher model max_new_tokens inference parameter.                                               | Yes      | 128     |
| `teacher_model_temperature`   | number    | Teacher model temperature inference parameter.                                                  | Yes      | 0.2     |
| `teacher_model_top_p`         | number    | Teacher model top_p inference parameter.                                                        | Yes      | 0.1     |
| `teacher_model_frequency_penalty` | number | Teacher model frequency penalty inference parameter.                                            | Yes      | 0.0     |
| `teacher_model_presence_penalty`  | number | Teacher model presence penalty inference parameter.                                             | Yes      | 0.0     |
| `teacher_model_stop`          | string    | Teacher model stop inference parameter.                                                         | No       |         |
| `enable_chain_of_thought`     | string    | Enable Chain of thought for data generation.                                                    | No       | "false" |
| `enable_chain_of_density`     | string    | Enable Chain of density for text summarization.                                                 | No       | "false" |
| `max_len_summary`             | integer   | Maximum Length Summary for text summarization.                                                  | No       | 80      |
| `data_generation_task_type`   | string    | Specifies the type of data generation task. Supported values: `NLI`, `CONVERSATION`, `NLU_QA`, `MATH`, `SUMMARIZATION`. | Yes      |         |
| `validation_output`           | uri_file  | Validation status.                                                                              | Yes      |         |

## Outputs
The component produces the following outputs:

| Name                             | Type      | Description                                                   |
|----------------------------------|-----------|---------------------------------------------------------------|
| `generated_train_payload_path`   | mltable   | Directory containing the payload to be sent to the model.     |
| `generated_validation_payload_path` | mltable | Directory containing the payload to be sent to the model.     |
| `hash_train_data`                | uri_file  | JSONL file containing the hash for each payload.              |
| `hash_validation_data`           | uri_file  | JSONL file containing the hash for each payload.              |
