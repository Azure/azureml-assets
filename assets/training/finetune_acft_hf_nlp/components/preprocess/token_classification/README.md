## Token Classification DataPreProcess

### Name 

token_classification_datapreprocess

### Version 

0.0.17

### Type 

command

### Description 

Component to preprocess data for token classification task. See [docs](https://aka.ms/azureml/components/token_classification_datapreprocess) to learn more.

## Inputs 

task arguments

sample input

{`tokens_column`: [ "EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "." ], `ner_tags_column`: '["B-ORG", "O", "B-MISC", "O", "O", "O", "B-MISC", "O", "O"]'}

For the above dataset pattern, `token_key` should be set as tokens_column and `tag_key` as ner_tags_column

| Name       | Description                                                          | Type    | Default | Optional | Enum |
| ---------- | -------------------------------------------------------------------- | ------- | ------- | -------- | ---- |
| token_key  | Key for tokens in each example line                                  | string  | -       | False    | NA   |
| tag_key    | Key for tags in each example line                                    | string  | -       | False    | NA   |
| batch_size | Number of examples to batch before calling the tokenization function | integer | 1000    | True     | NA   |



Tokenization params

pad_to_max_length:

type: string

enum:

- "true"

- "false"

default: "true"

optional: true

description: If set to True, the returned sequences will be padded according to the model's padding side and padding index, up to their `max_seq_length`. If no `max_seq_length` is specified, the padding is done up to the model's max length.

| Name           | Description                                                                                                          | Type    | Default | Optional | Enum |
| -------------- | -------------------------------------------------------------------------------------------------------------------- | ------- | ------- | -------- | ---- |
| max_seq_length | Default is -1 which means the padding is done up to the model's max length. Else will be padded to `max_seq_length`. | integer | -1      | True     | NA   |



Data inputs

Please note that either `train_file_path` or `train_mltable_path` needs to be passed. In case both are passed, `mltable path` will take precedence. The validation and test paths are optional and an automatic split from train data happens if they are not passed.

If both validation and test files are missing, 10% of train data will be assigned to each of them and the remaining 80% will be used for training

If anyone of the file is missing, 20% of the train data will be assigned to it and the remaining 80% will be used for training

| Name                    | Description                                                                                                               | Type     | Default | Optional | Enum |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------- | ------- | -------- | ---- |
| train_file_path         | Path to the registered training data asset. The supported data formats are `jsonl`, `json`, `csv`, `tsv` and `parquet`    | uri_file | -       | True     | NA   |
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