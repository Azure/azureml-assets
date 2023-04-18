# Text Classification DataPreProcess Component
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

    Output of Text Classification Model Import component

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

1. _sentence1_key_ (string, required)

    Key for `sentence1_key` in each example line

2. _sentence2_key_ (string, optional)

    Key for `sentence2_key` in each example line

3. _label_key_ (string, required)

    label key in each example line

4. _batch_size_ (int, optional)

    Number of examples to batch before calling the tokenization function. The default value is 1000.

Example1: Below is an example from CoLA train dataset

{"sentence":"Our friends won't buy this analysis, let alone the next one we propose.","label":true,"idx":0}

For this setting, `sentence1_key` is sentence, and `label_key` is label. The optional parameter `sentence2_key` can be ignored

Example2: Below is an example from MRPC train dataset

{"sentence1":"Amrozi accused his brother , whom he called \" the witness \" , of deliberately distorting his evidence .","sentence2":"Referring to him as only \" the witness \" , Amrozi accused his brother of deliberately distorting his evidence .","label":1,"idx":0}

If your dataset follows above pattern, `sentence1_key` should be set as sentence1 and `sentece2_key` as sentence2 `label_key` as label.
