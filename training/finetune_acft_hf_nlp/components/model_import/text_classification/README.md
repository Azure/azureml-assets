## Text Classification Model Import

### Name 

text_classification_model_import

### Version 

0.0.2

### Type 

command

### Description 

Component to import PyTorch / MLFlow model. See [docs](https://aka.ms/azureml/components/text_classification_model_import) to learn more.

## Inputs 

custom model id

| Name           | Description                                                                                                                                                                                                                                                                                                                                          | Type   | Default | Optional | Enum |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------- | -------- | ---- |
| huggingface_id | The string can be any Hugging Face id from the [Hugging Face models webpage](https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads). Models from Hugging Face are subject to third party license terms available on the Hugging Face model details page. It is your responsibility to comply with the model's license terms. | string | -       | True     | NA   |



PyTorch model as input

Folder structure for Pytorch model asset: The model folder is expected to contain model, config and tokenizer files and optionally optimizer, scheduler and the random states. The files are expected to be in the [Hugging Face format](https://huggingface.co/bert-base-uncased/tree/main). Additionally, the input folder **MUST** contain the file `finetune_args.json` with *model_name_or_path* as one of the keys of the dictionary. This file is already created if you are using an already finetuned model from Azureml

| Name               | Description              | Type         | Default | Optional | Enum |
| ------------------ | ------------------------ | ------------ | ------- | -------- | ---- |
| pytorch_model_path | Pytorch model asset path | custom_model | -       | True     | NA   |



MLflow model as an input

Folder structure for MLflow model asset:The model folder is expected to contain model, config and tokenizer files in a specific format as explained below -

- All the configuration files should be stored in _data/config_ folder

- All the model files should be stored in _data/model_ folder

- All the tokenizer files should be kept in _data/tokenizer_ folder

- **`MLmodel`** is a yaml file and this should contain _model_name_or_path_ information.

You could use the Model import pipeline to create a model of your own or refer to any of models in the Model Catalogue page if you want to manually create one. The MLflow output of a finetune model will be in correct format and no modification is needed.

| Name              | Description             | Type         | Default | Optional | Enum |
| ----------------- | ----------------------- | ------------ | ------- | -------- | ---- |
| mlflow_model_path | MLflow model asset path | mlflow_model | -       | True     | NA   |

## Outputs 

| Name       | Description                                                                                   | Type       |
| ---------- | --------------------------------------------------------------------------------------------- | ---------- |
| output_dir | Path to output directory which contains the component metadata and the model artifacts folder | uri_folder |