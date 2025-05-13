## Model Prediction Component

### Name 

validation_trigger_model_evaluation

### Version 

0.0.28

### Type 

command

### Description 

This component which validates inputs given to model evaluation pipeline by user.

## Inputs 


| Name               | Description                                                                         | Type    | Optional |
|--------------------| ----------------------------------------------------------------------------------- | ------- | ------- | 
| task               | Task type for which model is trained                                                                       | string  |  True     | 
| test_data          | Path to file containing test data in `jsonl` format | uri_file | True
| input_column_names | Name of the columns in the test dataset that should be used for prediction. More than one columns should be separated by the comma(,) delimiter without any whitespaces in between | string | True
| label_column_name  | Name of the key containing target values in test data. | string | True
| mlflow_model_path  |MLFlow model (either registered or output of another job) | mlflow_model | True
| device             | Option to run the experiment on CPU or GPU provided that the compute that they are choosing has Cuda support. | string | True
| batch_size         | Option to run the experiment on batch support. | integer | True
| evaluation_config          | Additional config file required by metrics package. This data asset should contain a JSON Config file. | uri_file    | True     |                                                |
| evaluation_config_params                       | JSON Serielized string of evaluation_config            | string | True                                                     |

## Outputs 

| Name                 | Description                                              | Type         |
| -------------------- | -------------------------------------------------------- | ------------ |
| validation_info | Validation status of the model evaluation pipeline inputs. | uri_file |