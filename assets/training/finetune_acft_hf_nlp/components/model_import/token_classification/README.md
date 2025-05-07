## Token Classification Model Import

### Name 

token_classification_model_import

### Version 

0.0.17

### Type 

command

### Description 

Component to import PyTorch / MLFlow model. See [docs](https://aka.ms/azureml/components/token_classification_model_import) to learn more.

## Inputs 

huggingface id

NOTE The pytorch_model_path or mlflow_model_path takes precedence over huggingface_id

| Name           | Description                                                                                                                                                                                                                                                                                                                                                 | Type   | Default | Optional | Enum |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------- | -------- | ---- |
| huggingface_id | The string can be any valid Hugging Face id from the [Hugging Face models webpage](https://huggingface.co/models?pipeline_tag=token-classification&sort=downloads). Models from Hugging Face are subject to third party license terms available on the Hugging Face model details page. It is your responsibility to comply with the model's license terms. | string | -       | True     | NA   |



PyTorch model as input

This is nothing but huggingface model folder. Here's the link to the example model folder - [bert-base-uncased](https://huggingface.co/bert-base-uncased/tree/main). Additionally, the model folder **MUST** contain the file `finetune_args.json` with *model_name_or_path* as one of the keys of the dictionary

| Name               | Description              | Type         | Default | Optional | Enum |
| ------------------ | ------------------------ | ------------ | ------- | -------- | ---- |
| pytorch_model_path | Pytorch model asset path | custom_model | -       | True     | NA   |



MLflow model as an input

This is also a huggingface model folder expect that the folder structure is slightly different. You could invoke a model import pipeline to convert the standard huggingface model into MLflow format. Please refer to this [notebook](https://aka.ms/azureml-import-model) for steps to do the same

NOTE The pytorch_model_path take priority over mlflow_model_path, in case both inputs are passed

| Name              | Description             | Type         | Default | Optional | Enum |
| ----------------- | ----------------------- | ------------ | ------- | -------- | ---- |
| mlflow_model_path | MLflow model asset path | mlflow_model | -       | True     | NA   |



Output of validation component

| Name              | Description        | Type     | Default | Optional | Enum |
| ----------------- | ------------------ | -------- | ------- | -------- | ---- |
| validation_output | Validation status. | uri_file | -       | True     | NA   |

## Outputs 

| Name       | Description                                                                                   | Type       |
| ---------- | --------------------------------------------------------------------------------------------- | ---------- |
| output_dir | Path to output directory which contains the component metadata and the model artifacts folder | uri_folder |