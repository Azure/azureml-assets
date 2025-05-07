## Text Generation Model Import

### Name 

text_generation_model_import

### Version 

0.0.17

### Type 

command

### Description 

Import PyTorch / MLFlow model

## Inputs 

Model and task name

| Name           | Description                                                                                                                                                                                                                                                           | Type   | Default | Optional | Enum |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------- | -------- | ---- |
| huggingface_id | Input HuggingFace model id. Incase of continual finetuning provide proper id. Models from Hugging Face are subject to third party license terms available on the Hugging Face model details page. It is your responsibility to comply with the model's license terms. | string | -       | True     | NA   |



Continual-Finetuning model path

| Name               | Description                                                                                                    | Type         | Default | Optional | Enum |
| ------------------ | -------------------------------------------------------------------------------------------------------------- | ------------ | ------- | -------- | ---- |
| pytorch_model_path | Input folder path containing pytorch model for further finetuning. Proper model/huggingface id must be passed. | custom_model | -       | True     | NA   |
| mlflow_model_path  | Input folder path containing mlflow model for further finetuning. Proper model/huggingface id must be passed.  | mlflow_model | -       | True     | NA   |

## Outputs 

| Name       | Description                    | Type       |
| ---------- | ------------------------------ | ---------- |
| output_dir | folder to store model metadata | uri_folder |