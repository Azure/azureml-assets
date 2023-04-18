# Translation DataPreProcess Component
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

    Output of Translation Model Import component

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

1. _source_lang_ (string, required)

    Key for source language in each example line. The key should be an abbreviated/coded form of the language as understood by tokenizer. For example, in case of __t5__ models, the `source_lang` should be set to "en" for English, "de" for German and "ro" for Romanian

    See the below table for list of supported source languages

    | __Model Family__ | Source Language (code)__ |
    | --- | --- |
    |t5| `["English (en)"]`|
    |mbart| `["Arabic (ar_AR)", "Czech (cs_CZ)", "German (de_DE)", "English (en_XX)", "Spanish (es_XX)", "Estonian (et_EE)", "Finnish (fi_FI)", "French (fr_XX)", "Gujarati (gu_IN)", "Hindi (hi_IN)", "Italian (it_IT)", "Japanese (ja_XX)", "Kazakh (kk_KZ)", "Korean (ko_KR)", "Lithuanian (lt_LT)", "Latvian (lv_LV)", "Burmese (my_MM)", "Nepali (ne_NP)", "Dutch (nl_XX)", "Romanian (ro_RO)", "Russian (ru_RU)", "Sinhala (si_LK)", "Turkish (tr_TR)", "Vietnamese (vi_VN)", "Chinese, Sim (zh_CN)"]`|

2. _target_lang_ (string, required)

    Key for translated language in each example line. The key should be an abbreviated/coded form of the language as understood by tokenizer. For example, in case of __t5__ models, the `source_lang` should be set to "en" for English, "de" for German and "ro" for Romanian

    See the below table for the list of supported target languages

    | __Model Family__ | __Target Language (code)__ |
    | --- | --- |
    |t5| `["French (fr)", "German (de)", "Romanian (ro)"]`|
    |mbart| `["Arabic (ar_AR)", "Czech (cs_CZ)", "German (de_DE)", "English (en_XX)", "Spanish (es_XX)", "Estonian (et_EE)", "Finnish (fi_FI)", "French (fr_XX)", "Gujarati (gu_IN)", "Hindi (hi_IN)", "Italian (it_IT)", "Japanese (ja_XX)", "Kazakh (kk_KZ)", "Korean (ko_KR)", "Lithuanian (lt_LT)", "Latvian (lv_LV)", "Burmese (my_MM)", "Nepali (ne_NP)", "Dutch (nl_XX)", "Romanian (ro_RO)", "Russian (ru_RU)", "Sinhala (si_LK)", "Turkish (tr_TR)", "Vietnamese (vi_VN)", "Chinese, Sim (zh_CN)"]`|

3. _batch_size_ (int, optional)

    Number of examples to batch before calling the tokenization function. The default value is 1000.

Example1: Below is one sample example from `wmt16 dataset` line that translates from English to Romanian,

{`en`:"Others have dismissed him as a joke.",`ro`:"Al\u021bii l-au numit o glum\u0103."}

If the dataset follows above pattern, `source_lang` is en and `target_lang` is ro
