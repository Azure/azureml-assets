## Common Validation Component

### Name 

ft_nlp_common_validation

### Version 

0.0.17

### Type 

command

### Description 

Component to validate the finetune job against Validation Service

## Inputs 

| Name              | Description                                                                                                                                                                                                                                                                                   | Type         | Default    | Optional | Enum |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ---------- | -------- | ---- |
| mlflow_model_path | MLflow model asset path. Special characters like \ and ' are invalid in the parameter value.                                                                                                                                                                                                  | mlflow_model | -          | True     | NA   |
| compute_finetune  | compute to be used for model_eavaluation eg. provide 'FT-Cluster' if your compute is named 'FT-Cluster'. Special characters like \ and ' are invalid in the parameter value. If compute cluster name is provided, instance_type field will be ignored and the respective cluster will be used | string       | serverless | True     | NA   |

## Outputs 

| Name            | Description        | Type     |
| --------------- | ------------------ | -------- |
| validation_info | Validation status. | uri_file |