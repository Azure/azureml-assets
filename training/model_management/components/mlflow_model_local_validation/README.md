## MLFlow model local validation

### Name 

mlflow_model_local_validation

### Version 

0.0.2

### Type 

command

### Description 

Validates if a MLFLow model can be loaded on a compute and is usable for inferencing.

## Inputs 

| Name              | Description                                                                                                            | Type         | Default | Optional | Enum |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------ | ------- | -------- | ---- |
| model_path        | MLFlow model to be validated                                                                                           | mlflow_model |         |          |      |
| test_data_path    | Test dataset for model inferencing                                                                                     | uri_file     |         | True     |      |
| column_rename_map | Provide mapping of dataset column names that should be renamed before inferencing. eg: col1:ren1; col2:ren2; col3:ren3 | string       |         | True     |      |

## Outputs 

| Name                | Description                                                                                                      | Type       |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------- |
| mlflow_model_folder | Validated input model. Here input model is used to block further steps in pipeline job if local validation fails | uri_folder |