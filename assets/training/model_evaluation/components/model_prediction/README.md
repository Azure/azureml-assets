## Model Prediction Component

### Name 

model_prediction

### Version 

0.0.17

### Type 

command

### Description 

This component which enables user to generate predictions on a given model.

## Inputs 


| Name               | Description                                                                         | Type    | Optional |
| ------------------ | ----------------------------------------------------------------------------------- | ------- | ------- | 
| task         | Task type for which model is trained                                                                       | string  |  True     | 
| test_data | Path to file containing test data in `jsonl` format | uri_file | True
| input_column_names | Name of the columns in the test dataset that should be used for prediction. More than one columns should be separated by the comma(,) delimiter without any whitespaces in between | string | True
| label_column_name | Name of the key containing target values in test data. | string | True
| mlflow_model |MLFlow model (either registered or output of another job) | mlflow_model | True
| device | Option to run the experiment on CPU or GPU provided that the compute that they are choosing has Cuda support. | string | True
| batch_size | Option to run the experiment on batch support. | integer | True

## Outputs 

| Name                 | Description                                              | Type         |
| -------------------- | -------------------------------------------------------- | ------------ |
| predictions | Predictions generated by the model | uri_file |
| prediction_probabilities | Prediction Probabilities generated by the model | uri_file
| ground_truth | Ground truth to evaluate predictions against | uri_file