## Question Answering DataPreProcess

### Name 

question_answering_datapreprocess

### Version 

0.0.17

### Type 

command

### Description 

Component to preprocess data for question answering task. See [docs](https://aka.ms/azureml/components/question_answering_datapreprocess) to learn more.

## Inputs 

Task arguments

Sample example

{`question_column`: "In what year did Paul VI formally appoint Mary as mother of the Catholic church?", `context_column`: "Paul VI opened the third period on 14 September 1964, telling the Council Fathers that he viewed the text about the Church as the most important document to come out from the Council. As the Council discussed the role of bishops in the papacy, Paul VI issued an explanatory note confirming the primacy of the papacy, a step which was viewed by some as meddling in the affairs of the Council American bishops pushed for a speedy resolution on religious freedom, but Paul VI insisted this to be approved together with related texts such as ecumenism. The Pope concluded the session on 21 November 1964, with the formal pronouncement of Mary as Mother of the Church.", `answers_column`: {`answer_start_column`: [595], `text_column`: ['1964']}}

If the dataset follows above pattern, `question_key`: "question_column"; `context_key`: "context_column"; `answers_key`: answers_column; `answer_start_key`: answer_start_column; `answer_text_key`: text_column

| Name                        | Description                                                                                                                                                                                | Type    | Default | Optional | Enum |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- | ------- | -------- | ---- |
| question_key                | The question whose answer needs to be extracted from the provided context                                                                                                                  | string  | -       | False    | NA   |
| context_key                 | The context that contains the answer to the question                                                                                                                                       | string  | -       | False    | NA   |
| answers_key                 | The value of this field is text in JSON format with two nested keys: answer_start_key and answer_text_key with their corresponding values                                                  | string  | -       | False    | NA   |
| answer_start_key            | Refers to the position where the answer beings in context. Needs a value that maps to a nested key in the values of the answers_key parameter                                              | string  | -       | False    | NA   |
| answer_text_key             | Contains the answer to the question. Needs a value that maps to a nested key in the values of the answers_key parameter                                                                    | string  | -       | False    | NA   |
| doc_stride                  | The amount of context overlap to keep in case the number of tokens per example exceed __max_seq_length__                                                                                   | integer | 128     | True     | NA   |
| n_best_size                 | The `top_n` max probable start tokens and end tokens to be consider while generating possible answers.                                                                                     | integer | 20      | True     | NA   |
| max_answer_length_in_tokens | The maximum allowed answer length specified in token length. The default value for this parameter is 30. All the answers with above 30 tokens will not be considered as a possible answer. | integer | 30      | True     | NA   |
| batch_size                  | Number of examples to batch before calling the tokenization function                                                                                                                       | integer | 1000    | True     | NA   |



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