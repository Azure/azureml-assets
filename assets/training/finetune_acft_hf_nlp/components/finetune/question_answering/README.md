## Question Answering Finetune

### Name 

question_answering_finetune

### Version 

0.0.17

### Type 

command

### Description 

Component to finetune Hugging Face pretrained models for extractive question answering task. The component supports optimizations such as LoRA, Deepspeed and ONNXRuntime for performance enhancement. See [docs](https://aka.ms/azureml/components/question_answering_finetune) to learn more.

## Inputs 

Lora parameters

LoRA reduces the number of trainable parameters by learning pairs of rank-decompostion matrices while freezing the original weights. This vastly reduces the storage requirement for large language models adapted to specific tasks and enables efficient task-switching during deployment all without introducing inference latency. LoRA also outperforms several other adaptation methods including adapter, prefix-tuning, and fine-tuning. Currently, LoRA is supported for gpt2, bert, roberta, deberta, distilbert, t5, bart, mbart and camembert model families

| Name               | Description                                                                                    | Type    | Default | Optional | Enum              |
| ------------------ | ---------------------------------------------------------------------------------------------- | ------- | ------- | -------- | ----------------- |
| apply_lora         | If "true" enables lora.                                                                        | string  | false   | True     | ['true', 'false'] |
| merge_lora_weights | If "true", the lora weights are merged with the base Hugging Face model weights before saving. | string  | true    | True     | ['true', 'false'] |
| lora_alpha         | alpha attention parameter for lora.                                                            | integer | 128     | True     | NA                |
| lora_r             | lora dimension                                                                                 | integer | 8       | True     | NA                |
| lora_dropout       | lora dropout value                                                                             | number  | 0.0     | True     | NA                |



Training parameters

| Name                        | Description                                                                                                                                                                                                                                                                           | Type    | Default  | Optional | Enum                                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | -------- | -------- | ---------------------------------------------------------------------------------------------- |
| num_train_epochs            | Number of epochs to run for finetune.                                                                                                                                                                                                                                                 | integer | 1        | True     | NA                                                                                             |
| max_steps                   | If set to a positive number, the total number of training steps to perform. Overrides 'epochs'. In case of using a finite iterable dataset the training may stop before reaching the set number of steps when all data is exhausted.                                                  | integer | -1       | True     | NA                                                                                             |
| per_device_train_batch_size | Per gpu batch size used for training. The effective training batch size is _per_device_train_batch_size_ * _num_gpus_ * _num_nodes_.                                                                                                                                                  | integer | 1        | True     | NA                                                                                             |
| per_device_eval_batch_size  | Per gpu batch size used for validation. The default value is 1. The effective validation batch size is _per_device_eval_batch_size_ * _num_gpus_ * _num_nodes_.                                                                                                                       | integer | 1        | True     | NA                                                                                             |
| auto_find_batch_size        | If set to "true" and if the provided 'per_device_train_batch_size' goes into Out Of Memory (OOM) auto_find_batch_size will find the correct batch size by iteratively reducing batch size by a factor of 2 till the OOM is fixed                                                      | string  | false    | True     | ['true', 'false']                                                                              |
| optim                       | Optimizer to be used while training                                                                                                                                                                                                                                                   | string  | adamw_hf | True     | ['adamw_hf', 'adamw_torch', 'adafactor']                                                       |
| learning_rate               | Start learning rate used for training.                                                                                                                                                                                                                                                | number  | 2e-05    | True     | NA                                                                                             |
| warmup_steps                | Number of steps for the learning rate scheduler warmup phase.                                                                                                                                                                                                                         | integer | 0        | True     | NA                                                                                             |
| weight_decay                | Weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer                                                                                                                                                                            | number  | 0.0      | True     | NA                                                                                             |
| adam_beta1                  | beta1 hyperparameter for the AdamW optimizer                                                                                                                                                                                                                                          | number  | 0.9      | True     | NA                                                                                             |
| adam_beta2                  | beta2 hyperparameter for the AdamW optimizer                                                                                                                                                                                                                                          | number  | 0.999    | True     | NA                                                                                             |
| adam_epsilon                | epsilon hyperparameter for the AdamW optimizer                                                                                                                                                                                                                                        | number  | 1e-08    | True     | NA                                                                                             |
| gradient_accumulation_steps | Number of updates steps to accumulate the gradients for, before performing a backward/update pass                                                                                                                                                                                     | integer | 1        | True     | NA                                                                                             |
| eval_accumulation_steps     | Number of predictions steps to accumulate before moving the tensors to the CPU                                                                                                                                                                                                        | integer | 1        | True     | NA                                                                                             |
| lr_scheduler_type           | learning rate scheduler to use.                                                                                                                                                                                                                                                       | string  | linear   | True     | ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'] |
| precision                   | Apply mixed precision training. This can reduce memory footprint by performing operations in half-precision.                                                                                                                                                                          | string  | 32       | True     | ['32', '16']                                                                                   |
| seed                        | Random seed that will be set at the beginning of training                                                                                                                                                                                                                             | integer | 42       | True     | NA                                                                                             |
| enable_full_determinism     | Ensure reproducible behavior during distributed training. Check this link https://pytorch.org/docs/stable/notes/randomness.html for more details.                                                                                                                                     | string  | false    | True     | ['true', 'false']                                                                              |
| dataloader_num_workers      | Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.                                                                                                                                                                             | integer | 0        | True     | NA                                                                                             |
| ignore_mismatched_sizes     | Not setting this flag will raise an error if some of the weights from the checkpoint do not have the same size as the weights of the model.                                                                                                                                           | string  | true     | True     | ['true', 'false']                                                                              |
| max_grad_norm               | Maximum gradient norm (for gradient clipping)                                                                                                                                                                                                                                         | number  | 1.0      | True     | NA                                                                                             |
| evaluation_strategy         | The evaluation strategy to adopt during training. If set to "steps", either the `evaluation_steps_interval` or `eval_steps` needs to be specified, which helps to determine the step at which the model evaluation needs to be computed else evaluation happens at end of each epoch. | string  | epoch    | True     | ['epoch', 'steps']                                                                             |
| evaluation_steps_interval   | The evaluation steps in fraction of an epoch steps to adopt during training. Overwrites eval_steps if not 0.                                                                                                                                                                          | number  | 0.0      | True     | NA                                                                                             |
| eval_steps                  | Number of update steps between two evals if evaluation_strategy='steps'                                                                                                                                                                                                               | integer | 500      | True     | NA                                                                                             |
| logging_strategy            | The logging strategy to adopt during training. If set to "steps", the `logging_steps` will decide the frequency of logging else logging happens at the end of epoch.                                                                                                                  | string  | epoch    | True     | ['epoch', 'steps']                                                                             |
| logging_steps               | Number of update steps between two logs if logging_strategy='steps'                                                                                                                                                                                                                   | integer | 500      | True     | NA                                                                                             |
| metric_for_best_model       | metric to use to compare two different model checkpoints                                                                                                                                                                                                                              | string  | loss     | True     | ['loss', 'f1', 'exact']                                                                        |
| resume_from_checkpoint      | If set to "true", resumes the training from last saved checkpoint. Along with loading the saved weights, saved optimizer, scheduler and random states will be loaded if exist. The default value is "false"                                                                           | string  | false    | True     | ['true', 'false']                                                                              |
| save_total_limit            | If a positive value is passed, it will limit the total number of checkpoints saved. The value of -1 saves all the checkpoints, otherwise if the number of checkpoints exceed the _save_total_limit_, the older checkpoints gets deleted.                                              | integer | -1       | True     | NA                                                                                             |



Early Stopping Parameters

| Name                     | Description                                                                                                                                                                                       | Type    | Default | Optional | Enum              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------- | -------- | ----------------- |
| apply_early_stopping     | If set to "true", early stopping is enabled.                                                                                                                                                      | string  | false   | True     | ['true', 'false'] |
| early_stopping_patience  | Stop training when the metric specified through _metric_for_best_model_ worsens for _early_stopping_patience_ evaluation calls.This value is only valid if _apply_early_stopping_ is set to true. | integer | 1       | True     | NA                |
| early_stopping_threshold | Denotes how much the specified metric must improve to satisfy early stopping conditions. This value is only valid if _apply_early_stopping_ is set to true.                                       | number  | 0.0     | True     | NA                |



Deepspeed Parameters

Deepspeed config is a JSON file that can be used to configure optimizer, scheduler, batch size and other training related parameters. A default deepspeed config is used when _apply_deepspeed_ is set to `true`. Alternatively, you can pass your custom deepspeed config. Please follow the [deepspeed docs](https://www.deepspeed.ai/docs/config-json/) to create the custom config.

Please note that to enable deepspeed, `apply_deepspeed` must be set to true, only passing the `deepspeed input` will not suffice

| Name            | Description                                        | Type     | Default | Optional | Enum              |
| --------------- | -------------------------------------------------- | -------- | ------- | -------- | ----------------- |
| apply_deepspeed | If set to true, will enable deepspeed for training | string   | false   | True     | ['true', 'false'] |
| deepspeed       | Deepspeed config to be used for finetuning         | uri_file | -       | True     | NA                |



ORT Parameters

ONNX Runtime is a cross-platform machine-learning model accelerator, with a flexible interface to integrate hardware-specific libraries.

| Name      | Description                                       | Type   | Default | Optional | Enum              |
| --------- | ------------------------------------------------- | ------ | ------- | -------- | ----------------- |
| apply_ort | If set to true, will use the ONNXRunTime training | string | false   | True     | ['true', 'false'] |



Data and Model inputs

| Name                  | Description                                                                                                                                   | Type       | Default | Optional | Enum |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------- | -------- | ---- |
| preprocess_output     | output folder of preprocess component containing encoded train, valid and test data. The tokenizer is also saved as part of preprocess output | uri_folder | -       | False    | NA   |
| model_selector_output | output folder of model import component containing model artifacts and a metadata file.                                                       | uri_folder | -       | False    | NA   |

## Outputs 

| Name                 | Description                                                                                                                                                                                                                                                                                                                           | Type         |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| pytorch_model_folder | output folder containing _best_ model as defined by _metric_for_best_model_. Along with the best model, output folder contains checkpoints saved after every evaluation which is defined by the _evaluation_strategy_. Each checkpoint contains the model weight(s), config, tokenizer, optimzer, scheduler and random number states. | uri_folder   |
| mlflow_model_folder  | output folder containing _best_ finetuned model in mlflow format.                                                                                                                                                                                                                                                                     | mlflow_model |