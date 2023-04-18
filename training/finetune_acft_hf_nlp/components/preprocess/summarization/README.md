# Summarization DataPreProcess Component
The goal of the component is to validate the user data and encode it. The encoded data along with relevant metadata is saved in the component output folder to be consumed by downstream components.

# 1. Inputs
1. _train_file_path_ (URI_FILE, optional)

    Path to the registered training data asset. The supported data formats are `jsonl`, `json`, `csv`, `tsv` and `parquet`.

2. _validation_file_path_ (URI_FILE, optional)

    Path to the registered validation data asset. The supported data formats are `jsonl`, `json`, `csv`, `tsv` and `parquet`.

3. _test_file_path_ (URI_FILE, optional)

    Path to the registered test data asset. The supported data formats are `jsonl`, `json`, `csv`, `tsv` and `parquet`.

4. _train_mltable_path_ (MLTABLE, optional)

    Path to the registered training data asset in `mltable` format.

5. _validation_mltable_path_ (MLTABLE, optional)

    Path to the registered validation data asset in `mltable` format.

6. _test_mltable_path_ (MLTABLE, optional)

    Registered test data asset in `mltable` format

7. _model_selector_output_ (URI_FOLDER, required)

    Output of Summarization Model Import component

> Please note that either `train_file_path` or `train_mltable_path` needs to be passed. In case both are passed, `mltable path` will take precedence. The validation and test paths are optional and an automatic split of train data happens if they are not passed. Below table shows the split details -

| __Validation file__ (could be mltable file or URI file) | __Test file__ (could be mltable file or URI file) | __Train Data Split__ (train-validation-test)|
| --- | --- | --- |
|missing|missing|80-10-10|
|missing|exists|80-0-20|
|exists|missing|80-20-0|

You can explore more about MLTable at [Working with tables in Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-mltable?tabs=cli%2Cpandas%2Cadls) and about its schema at [CLI (v2) mltable YAML schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-mltable).


# 2. Outputs
1. _output_dir_ (URI_FOLDER, required)

    The folder contains the tokenized output of the train, validation and test data along with the tokenizer files used to tokenize the data

# 3. Parameters

1. _document_key_ (string, required)

    Key for input document in each example line

2. _summary_key_ (string, required)

    Key for document summary in each example line

3. _summarization_lang_ (string, optional)

    Since, the current set of models `ONLY` support single language summarization, both the `document_key` and `summary_key` should be in same language.

    The parameter should be an abbreviated/coded form of the language as understood by tokenizer. Below table shows the list of supported languages for different model families

    | __Model Family__ | __Supported Language (code)__ |
    | --- | --- |
    |t5| `["French (fr)", "German (de)", "Romanian (ro)", "English (en)"]`|
    |mbart| `["Arabic (ar_AR)", "Czech (cs_CZ)", "German (de_DE)", "English (en_XX)", "Spanish (es_XX)", "Estonian (et_EE)", "Finnish (fi_FI)", "French (fr_XX)", "Gujarati (gu_IN)", "Hindi (hi_IN)", "Italian (it_IT)", "Japanese (ja_XX)", "Kazakh (kk_KZ)", "Korean (ko_KR)", "Lithuanian (lt_LT)", "Latvian (lv_LV)", "Burmese (my_MM)", "Nepali (ne_NP)", "Dutch (nl_XX)", "Romanian (ro_RO)", "Russian (ru_RU)", "Sinhala (si_LK)", "Turkish (tr_TR)", "Vietnamese (vi_VN)", "Chinese, Sim (zh_CN)"]`|
    |bart| `["English (en)"]`|

    Please note that the 2-letter code or 4-letter code needs to be used (mentioned in the brackets).

4. _batch_size_ (int, optional)

    Number of examples to batch before calling the tokenization function. The default value is 1000.

Example1: Below example is from xsum dataset

{`document_column`: "Cheryl Boone Isaacs said that the relationship with the accountancy firm PriceWaterhouseCoopers (PWC) was also under review.\nBrian Cullinan and Martha Ruiz were responsible for Sunday's mishap.\nLa La Land was mistakenly announced as the winner of the best picture award.\nThe team behind the film were in the middle of their speeches before it was revealed the accolade should have gone to Moonlight.\nIt has been described as the biggest mistake in 89 years of Academy Awards history.\nHow did the Oscars mistake happen?\nNine epic awards fails\nMr Cullinan mistakenly handed the wrong envelope to the two presenters.\nHe gave Warren Beatty and Faye Dunaway the back-up envelope for best actress in a leading role - rather than the envelope which contained the name of the winner for the best film.\nPriceWaterhouseCoopers, which counts the votes and organises the envelopes, has apologised for the mix-up.\nMr Cullinan tweeted a picture of best actress winner Emma Stone minutes before handing the presenters the wrong envelope, and Ms Boone Isaacs blamed "distraction" for the error.", `summary_column`: "The two accountants responsible for muddling up the main award envelopes at Sunday's Oscars ceremony will not be employed to do the job again, the academy president has announced."]}

For the above dataset pattern, `document_key` is document_column; `summary_key` is summary_column and `summarization_lang`: `en` for __t5__ family and `en_XX` for __mbart family__
