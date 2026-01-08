## Speculative Decoding Draft Pipeline

### Name

pipeline_draft_model

### Version

0.0.1

### Type

pipeline

### Description

Pipeline to train draft model for Speculative Decoding.

## Inputs

| Name                                      | Description                                                                                                                                                                                                     | Type    | Default            | Optional | Enum |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------------------ | -------- | ---- |
| instance_type_model_import                | Instance type to be used for model_import component in case of serverless compute, eg. standard_d12_v2. The parameter compute_model_import must be set to 'serverless' for instance_type to be used             | string  | Standard_d12_v2    | True     | NA   |
| instance_type_draft_model_training        | Instance type to be used for draft_model_training component in case of serverless compute, eg. standard_nc24rs_v3. The parameter compute_draft_training must be set to 'serverless' for instance_type to be used | string  | Standard_nc24rs_v3 | True     | NA   |
| num_nodes_draft_model_training            | number of nodes to be used for draft model training (used for distributed training)                                                                                                                             | integer | 1                  | True     | NA   |
| number_of_gpu_to_use_draft_model_training | number of gpus to be used per node for draft model training, should be equal to number of gpu per node in the compute SKU used for training                                                                     | integer | 1                  | True     | NA   |



Model Import parameters

| Name               | Description                                                                                                                                                                                                                                                                                                                                                                                                                   | Type         | Default | Optional | Enum |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------- | -------- | ---- |
| huggingface_id     | The string can be any valid Hugging Face id from the [Hugging Face models webpage](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads). Models from Hugging Face are subject to third party license terms available on the Hugging Face model details page. It is your responsibility to comply with the model's license terms. | string       | -       | True     | NA   |
| pytorch_model_path | Pytorch model asset path. Special characters like \ and ' are invalid in the parameter value.                                                                                                                                                                                                                                                                                                                                 | custom_model | -       | True     | NA   |
| mlflow_model_path  | MLflow model asset path. Special characters like \ and ' are invalid in the parameter value.                                                                                                                                                                                                                                                                                                                                  | mlflow_model | -       | True     | NA   |



Dataset parameters

| Name                       | Description                                           | Type     | Default | Optional | Enum |
| -------------------------- | ----------------------------------------------------- | -------- | ------- | -------- | ---- |
| dataset_train_split        | Path to the training dataset in JSONL format          | uri_file | -       | True     | NA   |
| dataset_validation_split   | Path to the validation dataset in JSONL format        | uri_file | -       | True     | NA   |



Draft Model Training parameters

| Name                       | Description                                                                                                                                                                                                           | Type     | Default         | Optional | Enum                                                            |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------------- | -------- | --------------------------------------------------------------- |
| draft_model_config         | Path to draft model configuration JSON file. If not provided, will be auto-generated from target model.                                                                                                               | uri_file | -               | True     | NA                                                              |
| num_epochs                 | Number of training epochs for draft model.                                                                                                                                                                            | integer  | 2               | True     | NA                                                              |
| training_batch_size        | Batch size for draft model training.                                                                                                                                                                                  | integer  | 2               | True     | NA                                                              |
| learning_rate              | Learning rate for draft model training.                                                                                                                                                                               | number   | 0.0001          | True     | NA                                                              |
| max_length                 | Maximum sequence length for draft model training.                                                                                                                                                                     | integer  | 2048            | True     | NA                                                              |
| warmup_ratio               | Warmup ratio for learning rate scheduler.                                                                                                                                                                             | number   | 0.015           | True     | NA                                                              |
| max_grad_norm              | Maximum gradient norm for gradient clipping.                                                                                                                                                                          | number   | 0.5             | True     | NA                                                              |
| ttt_length                 | The length for Test-Time Training (TTT).                                                                                                                                                                              | integer  | 7               | True     | NA                                                              |
| chat_template              | Chat template to use for formatting conversations. Supported templates - qwen, llama3, gpt-oss, deepseek-r1-distill.                                                                                                  | string   | llama3          | True     | ['llama3', 'qwen', 'gpt-oss', 'deepseek-r1-distill']            |
| attention_backend          | Attention implementation backend to use.                                                                                                                                                                              | string   | flex_attention  | True     | ['flex_attention', 'sdpa']                                      |
| tp_size                    | Tensor parallelism size                                                                                                                                                                                               | integer  | 1               | True     | NA                                                              |
| dp_size                    | Data parallelism size                                                                                                                                                                                                 | integer  | 1               | True     | NA                                                              |
| draft_global_batch_size    | Global batch size for draft model training                                                                                                                                                                            | integer  | 8               | True     | NA                                                              |
| draft_micro_batch_size     | Micro batch size for draft model                                                                                                                                                                                      | integer  | 1               | True     | NA                                                              |
| draft_accumulation_steps   | Gradient accumulation steps for draft model                                                                                                                                                                           | integer  | 1               | True     | NA                                                              |
| log_steps                  | Log training metrics every N steps                                                                                                                                                                                    | integer  | 50              | True     | NA                                                              |
| eval_interval              | Evaluation interval in epochs                                                                                                                                                                                         | integer  | 1               | True     | NA                                                              |
| save_interval              | Checkpoint save interval in epochs                                                                                                                                                                                    | integer  | 1               | True     | NA                                                              |
| seed                       | Random seed for reproducibility                                                                                                                                                                                       | integer  | 0               | True     | NA                                                              |
| total_steps                | Total training steps. If not provided, will be calculated as num_epochs * steps_per_epoch                                                                                                                             | integer  | -               | True     | NA                                                              |
| dist_timeout               | Timeout for collective communication in minutes                                                                                                                                                                       | integer  | 20              | True     | NA                                                              |
| resume                     | Whether to resume training from the last checkpoint                                                                                                                                                                   | string   | false           | True     | ['true', 'false']                                               |
| resume_from_checkpoint     | Path to a checkpoint directory to resume training from. Used when resume is true and no checkpoints exist in output folder, or when resume is false to initialize from a pretrained draft model checkpoint.          | uri_folder | -             | True     | NA                                                              |
| build_dataset_num_proc     | Number of processes to use for building the dataset. Recommended to set same as number of CPU cores available. If this is too high, one may loose performance due to context switching.                              | integer  | 96              | True     | NA                                                              |



Compute parameters

| Name                          | Description                                                                                                                                                                                                                                                                                   | Type   | Default    | Optional | Enum |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ---------- | -------- | ---- |
| compute_model_import          | compute to be used for model_import eg. provide 'FT-Cluster' if your compute is named 'FT-Cluster'. Special characters like \ and ' are invalid in the parameter value. If compute cluster name is provided, instance_type field will be ignored and the respective cluster will be used      | string | serverless | True     | NA   |
| compute_draft_model_training  | compute to be used for draft_model_training eg. provide 'FT-Cluster' if your compute is named 'FT-Cluster'. Special characters like \ and ' are invalid in the parameter value. If compute cluster name is provided, instance_type field will be ignored and the respective cluster will be used | string | serverless | True     | NA   |

## Outputs

| Name              | Description                                             | Type       |
| ----------------- | ------------------------------------------------------- | ---------- |
| output_model_path | Output folder containing trained draft model checkpoints. | uri_folder |
