## Common Model Converter

### Name 

ft_nlp_model_converter

### Version 

0.0.17

### Type 

command

### Description 

Component to convert the finetune job output to pytorch and mlflow model

## Inputs 

| Name              | Description                                                                                                                                                                                                                                                                                   | Type         | Default    | Optional | Enum |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ---------- | -------- | ---- |
| model_path | Pytorch model asset path. Pass the finetune job pytorch model output.                                                                                                                                                                                                  | uri_folder | -          | False     | NA   |
| converted_model  | Exisiting converted MlFlow path. Pass the finetune job mlflow model output. | mlflow_model       | - | True     | NA   |

## Outputs 

| Name            | Description        | Type     |
| --------------- | ------------------ | -------- |
| output_dir | Output folder containing _best_ finetuned model in mlflow format. | mlflow_model |