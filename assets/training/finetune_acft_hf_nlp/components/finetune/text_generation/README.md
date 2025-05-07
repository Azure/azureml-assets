## Text Generation Finetune

### Name 

text_generation_finetune

### Version 

0.0.17

### Type 

command

### Description 

Component to finetune model for Text Generation task

## Inputs 

Lora parameters

| Name               | Description                                                                         | Type    | Default | Optional | Enum              |
| ------------------ | ----------------------------------------------------------------------------------- | ------- | ------- | -------- | ----------------- |
| apply_lora         | lora enabled                                                                        | string  | false   | True     | ['true', 'false'] |
| merge_lora_weights | if set to true, the lora trained weights will be merged to base model before saving | string  | true    | True     | ['true', 'false'] |
| lora_alpha         | lora attention alpha                                                                | integer | 128     | True     | NA                |
| lora_r             | lora dimension                                                                      | integer | 8       | True     | NA                |
| lora_dropout       | lora dropout value                                                                  | number  | 0.0     | True     | NA                |



Training parameters

| Name                        | Description                                                                                                                                                                                                                                                                        | Type    | Default  | Optional | Enum                                                                                           |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | -------- | -------- | ---------------------------------------------------------------------------------------------- |
| num_train_epochs            | training epochs                                                                                                                                                                                                                                                                    | integer | 1        | True     | NA                                                                                             |
| max_steps                   | If set to a positive number, the total number of training steps to perform. Overrides 'epochs'. In case of using a finite iterable dataset the training may stop before reaching the set number of steps when all data is exhausted.                                               | integer | -1       | True     | NA                                                                                             |
| per_device_train_batch_size | Train batch size                                                                                                                                                                                                                                                                   | integer | 1        | True     | NA                                                                                             |
| per_device_eval_batch_size  | Validation batch size                                                                                                                                                                                                                                                              | integer | 1        | True     | NA                                                                                             |
| auto_find_batch_size        | Flag to enable auto finding of batch size. If the provided 'per_device_train_batch_size' goes into Out Of Memory (OOM) enabling auto_find_batch_size will find the correct batch size by iteratively reducing 'per_device_train_batch_size' by a factor of 2 till the OOM is fixed | string  | false    | True     | ['true', 'false']                                                                              |
| optim                       | Optimizer to be used while training                                                                                                                                                                                                                                                | string  | adamw_hf | True     | ['adamw_hf', 'adamw_torch', 'adafactor']                                                       |
| learning_rate               | Start learning rate. Defaults to linear scheduler.                                                                                                                                                                                                                                 | number  | 2e-05    | True     | NA                                                                                             |
| warmup_steps                | Number of steps used for a linear warmup from 0 to learning_rate                                                                                                                                                                                                                   | integer | 0        | True     | NA                                                                                             |
| weight_decay                | The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer                                                                                                                                                                     | number  | 0.0      | True     | NA                                                                                             |
| adam_beta1                  | The beta1 hyperparameter for the AdamW optimizer                                                                                                                                                                                                                                   | number  | 0.9      | True     | NA                                                                                             |
| adam_beta2                  | The beta2 hyperparameter for the AdamW optimizer                                                                                                                                                                                                                                   | number  | 0.999    | True     | NA                                                                                             |
| adam_epsilon                | The epsilon hyperparameter for the AdamW optimizer                                                                                                                                                                                                                                 | number  | 1e-08    | True     | NA                                                                                             |
| gradient_accumulation_steps | Number of updates steps to accumulate the gradients for, before performing a backward/update pass                                                                                                                                                                                  | integer | 1        | True     | NA                                                                                             |
| eval_accumulation_steps     | Number of predictions steps to accumulate before moving the tensors to the CPU                                                                                                                                                                                                     | integer | 1        | True     | NA                                                                                             |
| lr_scheduler_type           | learning rate scheduler to use.                                                                                                                                                                                                                                                    | string  | linear   | True     | ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'] |
| precision                   | Apply mixed precision training. This can reduce memory footprint by performing operations in half-precision.                                                                                                                                                                       | string  | 32       | True     | ['32', '16']                                                                                   |
| seed                        | Random seed that will be set at the beginning of training                                                                                                                                                                                                                          | integer | 42       | True     | NA                                                                                             |
| enable_full_determinism     | Ensure reproducible behavior during distributed training                                                                                                                                                                                                                           | string  | false    | True     | ['true', 'false']                                                                              |
| dataloader_num_workers      | Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.                                                                                                                                                                          | integer | 0        | True     | NA                                                                                             |
| ignore_mismatched_sizes     | Whether or not to raise an error if some of the weights from the checkpoint do not have the same size as the weights of the model                                                                                                                                                  | string  | true     | True     | ['true', 'false']                                                                              |
| max_grad_norm               | Maximum gradient norm (for gradient clipping)                                                                                                                                                                                                                                      | number  | 1.0      | True     | NA                                                                                             |
| evaluation_strategy         | The evaluation strategy to adopt during training                                                                                                                                                                                                                                   | string  | epoch    | True     | ['epoch', 'steps']                                                                             |
| evaluation_steps_interval   | The evaluation steps in fraction of an epoch steps to adopt during training. Overwrites evaluation_steps if not 0.                                                                                                                                                                 | number  | 0.0      | True     | NA                                                                                             |
| eval_steps                  | Number of update steps between two evals if evaluation_strategy='steps'                                                                                                                                                                                                            | integer | 500      | True     | NA                                                                                             |
| logging_strategy            | The logging strategy to adopt during training.                                                                                                                                                                                                                                     | string  | epoch    | True     | ['epoch', 'steps']                                                                             |
| logging_steps               | Number of update steps between two logs if logging_strategy='steps'                                                                                                                                                                                                                | integer | 500      | True     | NA                                                                                             |
| metric_for_best_model       | Specify the metric to use to compare two different models                                                                                                                                                                                                                          | string  | loss     | True     | ['loss']                                                                                       |
| resume_from_checkpoint      | Loads Optimizer, Scheduler and Trainer state for finetuning if true                                                                                                                                                                                                                | string  | false    | True     | ['true', 'false']                                                                              |
| save_total_limit            | If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir. If the value is -1 saves all checkpoints"                                                                                                                           | integer | -1       | True     | NA                                                                                             |



Early Stopping Parameters

| Name                     | Description                                                                                  | Type    | Default | Optional | Enum              |
| ------------------------ | -------------------------------------------------------------------------------------------- | ------- | ------- | -------- | ----------------- |
| apply_early_stopping     | Enable early stopping                                                                        | string  | false   | True     | ['true', 'false'] |
| early_stopping_patience  | Stop training when the specified metric worsens for early_stopping_patience evaluation calls | integer | 1       | True     | NA                |
| early_stopping_threshold | Denotes how much the specified metric must improve to satisfy early stopping conditions      | number  | 0.0     | True     | NA                |



Deepspeed Parameters

| Name            | Description                                        | Type     | Default | Optional | Enum              |
| --------------- | -------------------------------------------------- | -------- | ------- | -------- | ----------------- |
| apply_deepspeed | If set to true, will enable deepspeed for training | string   | false   | True     | ['true', 'false'] |
| deepspeed       | Deepspeed config to be used for finetuning         | uri_file | -       | True     | NA                |



ORT Parameters

| Name      | Description                                       | Type   | Default | Optional | Enum              |
| --------- | ------------------------------------------------- | ------ | ------- | -------- | ----------------- |
| apply_ort | If set to true, will use the ONNXRunTime training | string | false   | True     | ['true', 'false'] |



Dataset parameters

| Name                  | Description                                                                                            | Type       | Default | Optional | Enum |
| --------------------- | ------------------------------------------------------------------------------------------------------ | ---------- | ------- | -------- | ---- |
| preprocess_output     | output folder of preprocessor containing encoded train.jsonl valid.jsonl and the model pretrained info | uri_folder | -       | False    | NA   |
| model_selector_output | output folder of model selector containing model metadata like config, checkpoints, tokenizer config   | uri_folder | -       | False    | NA   |

## Outputs 

| Name                 | Description                                              | Type         |
| -------------------- | -------------------------------------------------------- | ------------ |
| pytorch_model_folder | Output dir to save the finetune model and other metadata | uri_folder   |
| mlflow_model_folder  | Output dir to save the finetune model as mlflow model    | mlflow_model |