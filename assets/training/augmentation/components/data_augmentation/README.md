## Data Generation Component

### Name

oss_augment_data

### Version

0.0.1

### Type

command

### Description

Component to augment data from synthetic data and original data

## Inputs

| Name                          | Description                                                 | Type     | Optional |
|-------------------------------| ----------------------------------------------------------- | -------  | -------  | 
| synthetic_data_file_path      | Path to the registered synthetic data set in `jsonl` format.| uri_file |  False   | 
| raw_data_file_path            | Path to the registered original data set in `jsonl` format. | uri_file |  False   |
| proportion_percentage         | Proportion of raw data to be mixed with Synthetic data.     | float    |  True    |
| seed                          | Teacher model endpoint URL. | string | True


## Outputs 

| Name                     | Description                                              | Type         |
| ------------------------ | -------------------------------------------------------- | ------------ |
| generated_data_file_path | Augmented data from synthetic and original data.         | uri_file     |
