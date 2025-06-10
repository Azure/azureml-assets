## Visual QnA DataPreProcess

### Name 

visual_qna_datapreprocess

### Version 

0.0.76

### Type 

command

### Description 

Component to preprocess data for visual qna task. 

## Inputs 

Task arguments

Sample example

# Visual Question Answering (VQA) Data Preprocessing

## Overview

This component preprocesses visual question answering data for fine-tuning multimodal language models. It handles image-text pairs where each question has multiple choice answers.

## Input Data Format

### Required JSONL Structure

Each line in the JSONL file should contain:

{
  "Figure_path": "path/to/image.jpg",
  "Question": "What color is the sky in this image?",
  "Choice A": "A. Blue",
  "Choice B": "B. Red", 
  "Choice C": "C. Green",
  "Choice D": "D. Yellow",
  "Answer": "A"
}

[ { Figure_path | Question | Choice A | Choice B | Choice C | Choice D | Answer

  } ]

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



Model input

| Name                  | Description                                                                                          | Type       | Default | Optional | Enum |
| --------------------- | ---------------------------------------------------------------------------------------------------- | ---------- | ------- | -------- | ---- |
| model_selector_output | output folder of model selector containing model metadata like config, checkpoints, tokenizer config | uri_folder | -       | False    | NA   |

## Outputs 

| Name       | Description                                                                                                                              | Type       |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| output_dir | The folder contains the tokenized output of the train, validation and test data along with the tokenizer files used to tokenize the data | uri_folder |