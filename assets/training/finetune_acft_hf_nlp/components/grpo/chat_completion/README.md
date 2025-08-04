## Group Relative Policy Optimization

### Name

group_relative_policy_optimization

### Version

0.0.1

### Type

command

### Description

Component to run Group Relative Policy optimization. Supports PyTorch distributed training with DeepSpeed optimizations.

## Inputs

Dataset and Model Inputs

| Name                      | Description                                                                                                      | Type       | Default | Optional | Enum |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------- | ------- | -------- | ---- |
| dataset_train_split       | Path to the training dataset in JSONL format                                                                    | uri_file   | -       | True     | NA   |
| dataset_validation_split  | Path to the validation dataset in JSONL format                                                                  | uri_file   | -       | True     | NA   |
| model_name_or_path        | output folder of model import component containing model artifacts and a metadata file.                         | uri_folder | -       | False    | NA   |
| deepspeed_config          | Path to a custom DeepSpeed configuration file in JSON format                                                    | uri_file   | -       | False    | NA   |
| dataset_name              | Name of the Hugging Face dataset to pull in                                                                     | string     | ''      | True     | NA   |
| dataset_prompt_column     | Column in the dataset containing the prompt for the chat completion template                                     | string     | problem | False    | NA   |

LoRA Parameters

| Name       | Description                       | Type   | Default | Optional | Enum              |
| ---------- | --------------------------------- | ------ | ------- | -------- | ----------------- |
| apply_lora | If "true" enables LoRA fine tuning. | string | true    | True     | ['true', 'false'] |

GRPO Algorithm Parameters

| Name             | Description                                                                                                                                   | Type    | Default | Optional | Enum |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------- | -------- | ---- |
| beta             | The beta parameter controls the strength of the KL divergence penalty in the objective function                                               | number  | 0.00    | True     | NA   |
| epsilon          | Epsilon value for clipping                                                                                                                    | number  | 0.5     | True     | NA   |
| num_iterations   | Number of iterations per batch (denoted as μ in the algorithm).                                                                               | integer | 3       | True     | NA   |
| num_generations  | Number of generations to sample.The effective batch size (num_processes*per_device_batch_size*gradient_accumulation_steps) must be evenly divisible by this value. | integer | 4       | True     | NA   |

Training Parameters

| Name                        | Description                                                                                                                                                                                        | Type    | Default      | Optional | Enum                                                                                                                            |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------- |
| num_train_epochs            | Number of training epochs.                                                                                                                                                                         | number  | 4            | True     | NA                                                                                                                              |
| max_steps                   | If set to a positive number, this will override num_train_epochs and train for exactly this many steps. Set to -1 to disable (default).                                                          | integer | -1           | True     | NA                                                                                                                              |
| per_device_train_batch_size | Per device batch size used for training                                                                                                                                                           | integer | 8            | True     | NA                                                                                                                              |
| per_device_eval_batch_size  | Per device batch size used for evaluation.                                                                                                                                                        | integer | 8            | True     | NA                                                                                                                              |
| gradient_accumulation_steps | Number of steps before performing a backward/update pass to accumulate gradients.                                                                                                                 | integer | 1            | True     | NA                                                                                                                              |
| learning_rate               | Learning rate for training.                                                                                                                                                                       | number  | 3e-06        | True     | NA                                                                                                                              |
| warmup_ratio                | Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.                                                                                                                | number  | 0.1          | True     | NA                                                                                                                              |
| max_grad_norm               | Maximum gradient norm for gradient clipping.                                                                                                                                                      | number  | 1.0          | True     | NA                                                                                                                              |
| optim                       | The optimizer to use.                                                                                                                                                                             | string  | adamw_torch  | True     | ['adamw_torch', 'adamw_torch_fused', 'adafactor', 'ademamix', 'sgd', 'adagrad', 'rmsprop', 'galore_adamw', 'lomo', 'adalomo', 'grokadamw', 'schedule_free_sgd'] |
| lr_scheduler_type           | The scheduler type to use for learning rate scheduling.                                                                                                                                           | string  | cosine       | True     | ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau'] |

Text Generation Parameters

| Name                 | Description                                                                                                     | Type    | Default | Optional | Enum |
| -------------------- | --------------------------------------------------------------------------------------------------------------- | ------- | ------- | -------- | ---- |
| max_prompt_length    | Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.               | integer | 512     | True     | NA   |
| max_completion_length| Maximum length of the generated completion.                                                                     | integer | 256     | True     | NA   |
| temperature          | Temperature for sampling. The higher the temperature, the more random the completions.                          | number  | 1.0     | True     | NA   |
| top_p                | Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to `1.0` to consider all tokens. | number  | 1.0     | True     | NA   |

Evaluation Parameters

| Name         | Description                                                                                                 | Type    | Default | Optional | Enum                 |
| ------------ | ----------------------------------------------------------------------------------------------------------- | ------- | ------- | -------- | -------------------- |
| do_eval      | Whether to run evaluation on the validation set or not. Will be set to True if eval_strategy is different from "no" | boolean | false   | True     | NA                   |
| eval_strategy| Evaluation strategy to use during training. Options are 'no', 'steps', or 'epoch'.                        | string  | no      | True     | ['no', 'steps', 'epoch'] |
| eval_steps   | Number of steps between evaluations                                                                         | integer | 1       | True     | NA                   |

Logging and Saving Parameters

| Name             | Description                                     | Type    | Default | Optional | Enum |
| ---------------- | ----------------------------------------------- | ------- | ------- | -------- | ---- |
| logging_steps    | Number of steps between logging updates.       | number  | 5       | True     | NA   |
| save_steps       | Number of steps between saving checkpoints.    | integer | 100     | True     | NA   |
| save_total_limit | Maximum number of checkpoints to keep.         | integer | 20      | True     | NA   |

Miscellaneous Parameters

| Name                        | Description                                           | Type    | Default | Optional | Enum |
| --------------------------- | ----------------------------------------------------- | ------- | ------- | -------- | ---- |
| shuffle_dataset             | Whether to shuffle the training dataset.             | boolean | true    | True     | NA   |
| use_liger_kernel            | Whether to use the Liger kernel.                     | boolean | false   | True     | NA   |
| vllm_gpu_memory_utilization | Control the GPU memory utilization for vLLM.         | number  | 0.3     | True     | NA   |
| vllm_tensor_parallel_size   | Control the tensor parallel size for vLLM.           | integer | 1       | True     | NA   |

## Outputs

| Name                 | Description                                                                                                                                                                                                                                                                                                                           | Type         |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| mlflow_model_folder  | output folder containing _best_ model as defined by _metric_for_best_model_. Along with the best model, output folder contains checkpoints saved after every evaluation which is defined by the _evaluation_strategy_. Each checkpoint contains the model weight(s), config, tokenizer, optimzer, scheduler and random number states. | mlflow_model |
| output_model_path    | Path to the output model folder containing the checkpoints                                                                                                                                                                                                                                                                            | uri_folder   |