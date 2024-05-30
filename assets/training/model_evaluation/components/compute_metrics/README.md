## Compute Metrics Component

### Name 

compute_metrics

### Version 

0.0.28

### Type 

command

### Description 

This component enables user to evaluate a model by providing generated predictions and true values to return generated metrics. (Scores the predictions provided by user. No model is required in this case).

## Inputs 


| Name               | Description                                                                         | Type    | Optional |
| ------------------ | ----------------------------------------------------------------------------------- | ------- | ------- | 
| task         | Task type for which model is trained                                                                       | string  |  True     | 
| ground_truth | Actual ground truth to evaluate predictions against. The file should be of JSON lines format containing only one key. | uri_file  |  True     | 
| ground_truth_column_name             | Column name which contains ground truths in provided uri file for ground_truths.                                                                    | string | True     |                   |
| prediction      | Actual predictions which are to be evaluated. They should be in json lines too with only one key.                                                                  | uri_file  | True     |                   |
| prediction_column_name                 | Column name which contains predictions in provided uri file for predictions.                                               | string   | True     |                                                                                                |
| prediction_probabilites | Prediction probabilities in order to calculate better set of metrics for classification tasks. This file should be in JSON lines format as well with number of keys equals to number of unique labels.                         | uri_file      | True     |                                                                                                |
| evaluation_config          | Additional config file required by metrics package. This data asset should contain a JSON Config file. | uri_file    | True     |                                                |
| evaluation_config_params                       | JSON Serielized string of evaluation_config            | string | True                                                     |

## Outputs 

| Name                 | Description                                              | Type         |
| -------------------- | -------------------------------------------------------- | ------------ |
| evaluationResult | Output dir to save the finetune model and other metadata | uri_folder   |
