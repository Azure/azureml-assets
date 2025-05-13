## Evaluate Model Component

### Name 

evaluate_model

### Version 

0.0.27

### Type 

command

### Description 

This component enables user to evaluate a model by providing the supported model, run inference to generate predictions first followed by computing metrics against a dataset. You can find the component in your workspace components page.

## Inputs 


| Name               | Description                                                                         | Type    | Optional |
| ------------------ | ----------------------------------------------------------------------------------- | ------- | ------- | 
| task         | Task type for which model is trained                                                                       | string  |  True     | 
| test_data | Path to file containing test data in `jsonl` format | uri_file | True
| test_data_mltable | Test data in the form of mltables | ml_table | True
| test_data_input_column_names | Name of the columns in the test dataset that should be used for prediction. More than one columns should be separated by the comma(,) delimiter without any whitespaces in between | string | True
| test_data_label_column_name | Name of the key containing target values in test data. | string | True
| mlflow_model |MLFlow model (either registered or output of another job) | mlflow_model | True
| model_uri |  MLFlow model uri of the form - <br> fetched from azureml run as `runs:/<azureml_run_id>/run-relative/path/to/model` <br> fetched from azureml model registry as `models:/<model_name>/<model_version>` | string | True
| evaluation_config          | Additional config file required by metrics package. This data asset should contain a JSON Config file. | uri_file    | True     |                                                |
| evaluation_config_params                       | JSON Serielized string of evaluation_config            | string | True                                                     |
| device | Option to run the experiment on CPU or GPU provided that the compute that they are choosing has Cuda support. | string | True
| batch_size | Option to run the experiment on batch support. | integer | True

## Outputs 

| Name                 | Description                                              | Type         |
| -------------------- | -------------------------------------------------------- | ------------ |
| evaluationResult | Output dir to save the finetune model and other metadata | uri_folder   |
