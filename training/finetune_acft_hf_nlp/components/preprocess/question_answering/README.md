# Question Answering DataPreProcess Component
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

    Output of Question Answering Model Import component

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

1. _question_key_ (string, required)

    Key for question in each example line

2. _context_key_ (string, required)

    Key for context in each example line

3. _answers_key_ (string, required)

    Key for answers in each example line

4. _answer_start_key_ (int, required)

    Key for answer start in each example line

5. _answer_text_key_ (string, required)

    Key for answer text in each example line

6. _doc_stride_ (int, optional)

    The amount of context overlap to keep in case the number of tokens per example exceed __max_seq_length__. The default value for the paramter is 128.

7. _n_best_size_ (int, optional)

    The `top_n` max probable start tokens and end tokens to be consider while generating possible answers. The default value for the parameter is 20.

8. _max_answer_length_in_tokens_ (int, optional)

    The maximum allowed answer length specified in token length. The default value for this parameter is 30. All the answers with above 30 tokens will not be considered as a possible answer.

9. _batch_size_ (int, optional)

    Number of examples to batch before calling the tokenization function. The default value is 1000.

Example1: Below is the sample example from squad dataset,

{`question_column`: "In what year did Paul VI formally appoint Mary as mother of the Catholic church?", `context_column`: "Paul VI opened the third period on 14 September 1964, telling the Council Fathers that he viewed the text about the Church as the most important document to come out from the Council. As the Council discussed the role of bishops in the papacy, Paul VI issued an explanatory note confirming the primacy of the papacy, a step which was viewed by some as meddling in the affairs of the Council American bishops pushed for a speedy resolution on religious freedom, but Paul VI insisted this to be approved together with related texts such as ecumenism. The Pope concluded the session on 21 November 1964, with the formal pronouncement of Mary as Mother of the Church.", `answers_column`: {`answer_start_column`: [595], `text_column`: ['1964']}}

If the dataset follows above pattern, `question_key`: "question_column"; `context_key`: "context_column"; `answers_key`: answers_column; `answer_start_key`: answer_start_column; `answer_text_key`: text_column
