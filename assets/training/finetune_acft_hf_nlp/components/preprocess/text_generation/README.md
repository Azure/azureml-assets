## Text Generation DataPreProcess

### Name 

text_generation_datapreprocess

### Version 

0.0.17

### Type 

command

### Description 

Component to preprocess data for text generation task

## Inputs 

Text Generation task arguments

| Name             | Description                                                                                                                                                                                                                                                                                                                                                                                                                     | Type    | Default | Optional | Enum |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------- | -------- | ---- |
| text_key         | key for text in an example. format your data keeping in mind that text is concatenated with ground_truth while finetuning in the form - text + groundtruth. for eg. "text"="knock knock\n", "ground_truth"="who's there"; will be treated as "knock knock\nwho's there"                                                                                                                                                         | string  | -       | False    | NA   |
| ground_truth_key | key for ground_truth in an example. we take separate column for ground_truth to enable use cases like summarization, translation, question_answering, etc. which can be repurposed in form of text-generation where both text and ground_truth are needed. This separation is useful for calculating metrics. for eg. "text"="Summarize this dialog:\n{input_dialogue}\nSummary:\n", "ground_truth"="{summary of the dialogue}" | string  | -       | False    | NA   |
| batch_size       | Number of examples to batch before calling the tokenization function                                                                                                                                                                                                                                                                                                                                                            | integer | 1000    | True     | NA   |



Tokenization params

| Name              | Description                                                                                                                                                                                                                         | Type    | Default | Optional | Enum              |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------- | -------- | ----------------- |
| pad_to_max_length | If set to True, the returned sequences will be padded according to the model's padding side and padding index, up to their `max_seq_length`. If no `max_seq_length` is specified, the padding is done up to the model's max length. | string  | false   | True     | ['true', 'false'] |
| max_seq_length    | Default is -1 which means the padding is done up to the model's max length. Else will be padded to `max_seq_length`.                                                                                                                | integer | -1      | True     | NA                |



Inputs

| Name                    | Description                       | Type     | Default | Optional | Enum |
| ----------------------- | --------------------------------- | -------- | ------- | -------- | ---- |
| train_file_path         | Enter the train file path         | uri_file | -       | True     | NA   |
| validation_file_path    | Enter the validation file path    | uri_file | -       | True     | NA   |
| test_file_path          | Enter the test file path          | uri_file | -       | True     | NA   |
| train_mltable_path      | Enter the train mltable path      | mltable  | -       | True     | NA   |
| validation_mltable_path | Enter the validation mltable path | mltable  | -       | True     | NA   |
| test_mltable_path       | Enter the test mltable path       | mltable  | -       | True     | NA   |



Dataset parameters

| Name                  | Description                                                                                          | Type       | Default | Optional | Enum |
| --------------------- | ---------------------------------------------------------------------------------------------------- | ---------- | ------- | -------- | ---- |
| model_selector_output | output folder of model selector containing model metadata like config, checkpoints, tokenizer config | uri_folder | -       | False    | NA   |

## Outputs 

| Name       | Description                                        | Type       |
| ---------- | -------------------------------------------------- | ---------- |
| output_dir | folder to store preprocessed outputs of input data | uri_folder |