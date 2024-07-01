## Model Evaluation Pipeline Component

### Name 

model_evaluation_pipeline_component

### Version 

0.0.28

### Type 

pipeline

### Description 

This pipeline component for model evaluation for supported tasks. Generates predictions on a given model, followed by computing model performance metrics to score the model quality for supported tasks.

## Inputs 


| Name               | Description                                                                         | Type    | Optional |
| ------------------ | ----------------------------------------------------------------------------------- | ------- | ------- | 
| task         | Task type for which model is trained                                                                       | string  |  True     | 
| test_data | Path to file containing test data in `jsonl` format | uri_file | True
| input_column_names | Name of the columns in the test dataset that should be used for prediction. More than one columns should be separated by the comma(,) delimiter without any whitespaces in between | string | True
| label_column_name | Name of the key containing target values in test data. | string | True
| mlflow_model |MLFlow model (either registered or output of another job) | mlflow_model | True
| evaluation_config          | Additional config file required by metrics package. This data asset should contain a JSON Config file. | uri_file    | True     |                                                |
| evaluation_config_params                       | JSON Serielized string of evaluation_config            | string | True                                                     |
| device | Option to run the experiment on CPU or GPU provided that the compute that they are choosing has Cuda support. | string | True
| batch_size | Option to run the experiment on batch support. | integer | True

## Outputs 

| Name                 | Description                                              | Type         |
| -------------------- | -------------------------------------------------------- | ------------ |
| evaluationResult | Output dir to save the finetune model and other metadata | uri_folder   |
