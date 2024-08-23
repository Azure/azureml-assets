## Data Generation Component

### Name

oss_distillation_generate_data

### Version

0.0.4

### Type

command

### Description

Component to generate data from teacher model enpoint

## Inputs

| Name               | Description                                                                         | Type    | Optional |
|--------------------| ----------------------------------------------------------------------------------- | ------- | ------- | 
| train_file_path               | Path to the registered training data set in `jsonl, json, csv, tsv and parquet` format. | uri_file  |  True     | 
| validation_file_path          | Path to the registered training data set in `jsonl, json, csv, tsv and parquet` format. | uri_file | True
| teacher_model_endpoint_name | Teacher model endpoint name. | string | True
| teacher_model_endpoint_url  | Teacher model endpoint URL. | string | True
| teacher_model_endpoint_key  | Teacher model endpoint key. | string | True
| teacher_model_max_new_tokens | Teacher model max_new_tokens inference parameter. | integer | True
| teacher_model_temperature   | Teacher model temperature inference parameter. | number | True
| teacher_model_top_p          | Teacher model top_p inference parameter. | number    | True     |  |
| teacher_model_frequency_penalty  | Teacher model frequency penalty inference parameter.  | number | True  |
| teacher_model_presence_penalty | Teacher model presence penalty inference parameter. | number | True
| teacher_model_stop | Teacher model stop inference parameter. | string | True
| request_batch_size | No of data records to hit teacher model endpoint in one go. | integer | True
| min_endpoint_success_ratio | The minimum value of (successful_requests / total_requests) required for classifying inference as successful. | number | True
| enable_chain_of_thought | Enable Chain of thought for data generation. | string | True
| data_generation_task_type | Data generation task types, supported values - NLI, CONVERSATION, NLU_QA. | string | False
| validation_output | Validation status from validation component. | uri_file | True

## Outputs 

| Name                 | Description                                              | Type         |
| -------------------- | -------------------------------------------------------- | ------------ |
| generated_train_file_path | Generated training data. | uri_file |
| generated_validation_file_path | Generated validation data. | uri_file |