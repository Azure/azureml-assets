## Download model

### Name 

download_model

### Version 

0.0.3

### Type 

command

### Description 

Downloads a publicly available model.

## Inputs 

| Name         | Description                                                                                                                                                                                                                                  | Type   | Default     | Optional | Enum                                |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ----------- | -------- | ----------------------------------- |
| model_source | Storage containers from where model will be sourced from.                                                                                                                                                                                    | string | Huggingface |          | ['AzureBlob', 'GIT', 'Huggingface'] |
| model_id     | A valid model id for the model source selected. For example you can specify `bert-base-uncased` for importing HuggingFace bert base uncased model. Please specify the complete URL if **GIT** or **AzureBlob** is selected in `model_source` | string |             |          |                                     |

## Outputs 

| Name                    | Description                                                                                                                                                             | Type       |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| model_download_metadata | File name to which model download details will be written. File would contain details that could be useful for model registration in forms of model tags and properties | uri_file   |
| model_output            | Path to the dowloaded model                                                                                                                                             | uri_folder |