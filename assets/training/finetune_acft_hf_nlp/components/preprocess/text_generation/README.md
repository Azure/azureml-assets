## Text Generation DataPreProcess

### Name 

text_generation_datapreprocess

### Version 

0.0.11

### Type 

command

### Description 

Component to preprocess data for text generation task. See [docs](https://aka.ms/azureml/components/text_generation_datapreprocess) to learn more.

## Inputs 

task arguments

sample input

{`text`:"Others have dismissed him as a joke."}

| Name       | Description                                                          | Type    | Default | Optional | Enum |
| ---------- | -------------------------------------------------------------------- | ------- | ------- | -------- | ---- |
| text_key   | key for text in an example                                           | string  | -       | False    | NA   |
| batch_size | Number of examples to batch before calling the tokenization function | integer | 1000    | True     | NA   |



Tokenization params

| Name              | Description                                                                                          | Type    | Default | Optional | Enum |
| ----------------- | ---------------------------------------------------------------------------------------------------- | ------- | ------- | -------- | ---- |
| pad_to_max_length | output folder of model selector containing model metadata like config, checkpoints, tokenizer config | boolean | True    | True     | NA   |
| max_seq_length    | output folder of model selector containing model metadata like config, checkpoints, tokenizer config | integer | -1      | True     | NA   |



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