## Text Classification DataPreProcess

### Name 

text_classification_datapreprocess

### Version 

0.0.17

### Type 

command

### Description 

Component to preprocess data for single label classification task. See [docs](https://aka.ms/azureml/components/text_classification_datapreprocess) to learn more.

## Inputs 

Text Claasification task type

| Name      | Description                   | Type   | Default                   | Optional | Enum                          |
| --------- | ----------------------------- | ------ | ------------------------- | -------- | ----------------------------- |
| task_name | Text Classification task type | string | SingleLabelClassification | False    | ['SingleLabelClassification'] |



task arguments

sample input1

{"sentence":"Our friends won't buy this analysis, let alone the next one we propose.","label":true,"idx":0}

For this setting, `sentence1_key` is sentence, and `label_key` is label. The optional parameter `sentence2_key` can be ignored

sample input2

{"sentence1":"Amrozi accused his brother , whom he called \" the witness \" , of deliberately distorting his evidence .","sentence2":"Referring to him as only \" the witness \" , Amrozi accused his brother of deliberately distorting his evidence .","label":1,"idx":0}

If your dataset follows above pattern, `sentence1_key` should be set as sentence1 and `sentence2_key` as sentence2 `label_key` as label.

| Name          | Description                                                          | Type    | Default | Optional | Enum |
| ------------- | -------------------------------------------------------------------- | ------- | ------- | -------- | ---- |
| sentence1_key | Key for `sentence1_key` in each example line                         | string  | -       | False    | NA   |
| sentence2_key | Key for `sentence2_key` in each example line                         | string  | -       | True     | NA   |
| label_key     | label key in each example line                                       | string  | -       | False    | NA   |
| batch_size    | Number of examples to batch before calling the tokenization function | integer | 1000    | True     | NA   |



Tokenization params

| Name              | Description                                                                                                                                                                                                                         | Type    | Default | Optional | Enum              |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------- | -------- | ----------------- |
| pad_to_max_length | If set to True, the returned sequences will be padded according to the model's padding side and padding index, up to their `max_seq_length`. If no `max_seq_length` is specified, the padding is done up to the model's max length. | string  | false   | True     | ['true', 'false'] |
| max_seq_length    | Controls the maximum length to use when pad_to_max_length parameter is set to `true`. Default is -1 which means the padding is done up to the model's max length. Else will be padded to `max_seq_length`.                          | integer | -1      | True     | NA                |



Data inputs

Please note that either `train_file_path` or `train_mltable_path` needs to be passed. In case both are passed, `mltable path` will take precedence. The validation and test paths are optional and an automatic split from train data happens if they are not passed.

If both validation and test files are missing, 10% of train data will be assigned to each of them and the remaining 80% will be used for training

If anyone of the file is missing, 20% of the train data will be assigned to it and the remaining 80% will be used for training

| Name                    | Description                                                                                                               | Type     | Default | Optional | Enum |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------- | ------- | -------- | ---- |
| train_file_path         | Path to the registered training data asset. The supported data formats are `jsonl`, `json`, `csv`, `tsv` and `parquet`.   | uri_file | -       | True     | NA   |
| validation_file_path    | Path to the registered validation data asset. The supported data formats are `jsonl`, `json`, `csv`, `tsv` and `parquet`. | uri_file | -       | True     | NA   |
| test_file_path          | Path to the registered test data asset. The supported data formats are `jsonl`, `json`, `csv`, `tsv` and `parquet`.       | uri_file | -       | True     | NA   |
| train_mltable_path      | Path to the registered training data asset in `mltable` format.                                                           | mltable  | -       | True     | NA   |
| validation_mltable_path | Path to the registered validation data asset in `mltable` format.                                                         | mltable  | -       | True     | NA   |
| test_mltable_path       | Path to the registered test data asset in `mltable` format.                                                               | mltable  | -       | True     | NA   |



Dataset parameters

| Name                  | Description                                                                                          | Type       | Default | Optional | Enum |
| --------------------- | ---------------------------------------------------------------------------------------------------- | ---------- | ------- | -------- | ---- |
| model_selector_output | output folder of model selector containing model metadata like config, checkpoints, tokenizer config | uri_folder | -       | False    | NA   |

## Outputs 

| Name       | Description                                                                                                                              | Type       |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| output_dir | The folder contains the tokenized output of the train, validation and test data along with the tokenizer files used to tokenize the data | uri_folder |