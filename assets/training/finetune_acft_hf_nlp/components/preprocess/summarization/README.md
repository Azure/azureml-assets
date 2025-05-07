## Summarization DataPreProcess

### Name 

summarization_datapreprocess

### Version 

0.0.17

### Type 

command

### Description 

Component to preprocess data for summarization task. See [docs](https://aka.ms/azureml/components/summarization_datapreprocess) to learn more.

## Inputs 

Task arguments

Sample example

{`document_column`: "Cheryl Boone Isaacs said that the relationship with the accountancy firm PriceWaterhouseCoopers (PWC) was also under review.\nBrian Cullinan and Martha Ruiz were responsible for Sunday's mishap.\nLa La Land was mistakenly announced as the winner of the best picture award.\nThe team behind the film were in the middle of their speeches before it was revealed the accolade should have gone to Moonlight.\nIt has been described as the biggest mistake in 89 years of Academy Awards history.\nHow did the Oscars mistake happen?\nNine epic awards fails\nMr Cullinan mistakenly handed the wrong envelope to the two presenters.\nHe gave Warren Beatty and Faye Dunaway the back-up envelope for best actress in a leading role - rather than the envelope which contained the name of the winner for the best film.\nPriceWaterhouseCoopers, which counts the votes and organises the envelopes, has apologised for the mix-up.\nMr Cullinan tweeted a picture of best actress winner Emma Stone minutes before handing the presenters the wrong envelope, and Ms Boone Isaacs blamed "distraction" for the error.", `summary_column`: "The two accountants responsible for muddling up the main award envelopes at Sunday's Oscars ceremony will not be employed to do the job again, the academy president has announced."]}

For the above dataset pattern, `document_key` is document_column; `summary_key` is summary_column and `summarization_lang`: `en` for __t5__ family and `en_XX` for __mbart family__

summarization_lang codes for T5, mbart and bart

t5 - French (fr), German (de), Romanian (ro), English (en)

mbart - Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese, Sim (zh_CN)

bart - English (en)

| Name               | Description                                                                                                                                                                      | Type    | Default | Optional | Enum |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------- | -------- | ---- |
| document_key       | Key for input document in each example line                                                                                                                                      | string  | -       | False    | NA   |
| summary_key        | Key for document summary in each example line                                                                                                                                    | string  | -       | False    | NA   |
| summarization_lang | The parameter should be an abbreviated/coded form of the language as understood by tokenizer. Please check the respective model's language codes while updating this information | string  | -       | True     | NA   |
| batch_size         | Number of examples to batch before calling the tokenization function                                                                                                             | integer | 1000    | True     | NA   |



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