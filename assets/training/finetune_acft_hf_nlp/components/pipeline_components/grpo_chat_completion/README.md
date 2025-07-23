## GRPO Chat Completion Pipeline

### Name

grpo_chat_completion_pipeline

### Version

0.0.1

### Type

pipeline

### Description

Pipeline component for fine-tuning Hugging Face chat completion models with Group Relative Policy Optimization(GRPO)

## Inputs

### Infrastructure Parameters

| Name                            | Description                                                                                                                                                                                     | Type    | Default            | Optional |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------------------ | -------- |
| instance_type_model_import      | Instance type to be used for model_import component in case of serverless compute, eg. standard_d12_v2. The parameter compute_model_import must be set to 'serverless' for instance_type to be used | string  | Standard_d12_v2    | True     |
| instance_type_finetune          | Instance type to be used for finetune component in case of serverless compute, eg. standard_nc24rs_v3. The parameter compute_finetune must be set to 'serverless' for instance_type to be used     | string  | STANDARD_ND96ISRF_H100_V5 | True     |
| shm_size_finetune               | Shared memory size to be used for finetune component. It is useful while using Nebula (via DeepSpeed) which uses shared memory to save model and optimizer states.                            | string  | 5g                 | True     |
| num_nodes_finetune              | number of nodes to be used for finetuning (used for distributed training)                                                                                                                      | integer | 1                  | True     |
| number_of_gpu_to_use_finetuning | number of gpus to be used per node for finetuning, should be equal to number of gpu per node in the compute SKU used for finetune                                                              | integer | 1                  | True     |

### Model Import Parameters

| Name               | Description                                                                                                                                                                                                                                                                                                                                                                                                                   | Type         | Default | Optional |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------- | -------- |
| huggingface_id     | The string can be any valid Hugging Face id from the [Hugging Face models webpage](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads). Models from Hugging Face are subject to third party license terms available on the Hugging Face model details page. It is your responsibility to comply with the model's license terms. | string       | -       | True     |
| pytorch_model_path | Pytorch model asset path. Special characters like \ and ' are invalid in the parameter value.                                                                                                                                                                                                                                                                                                                                 | custom_model | -       | True     |
| mlflow_model_path  | MLflow model asset path. Special characters like \ and ' are invalid in the parameter value.                                                                                                                                                                                                                                                                                                                                  | mlflow_model | -       | True     |

### Dataset Parameters

| Name                       | Description                                                              | Type     | Default | Optional |
| -------------------------- | ------------------------------------------------------------------------ | -------- | ------- | -------- |
| dataset_name               | Name of the Hugging Face dataset to pull in                             | string   | ''      | True     |
| dataset_prompt_column      | Column in the dataset containing the prompt for the chat completion template | string   | problem | False    |
| dataset_train_split        | Path to the training dataset in JSONL format                            | uri_file | -       | True     |
| dataset_validation_split   | Path to the validation dataset in JSONL format                          | uri_file | -       | True     |

### Training Parameters

| Name                        | Description                                                                                           | Type    | Default     | Optional | Enum                                                                                                                                                  |
| --------------------------- | ----------------------------------------------------------------------------------------------------- | ------- | ----------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| apply_lora                  | If "true" enables lora.                                                                              | string  | true        | True     | ['true', 'false']                                                                                                                                     |
| eval_strategy               | Evaluation strategy to use during training. Options are 'no', 'steps', or 'epoch'.                  | string  | no          | True     | ['no', 'steps', 'epoch']                                                                                                                              |
| num_iterations              | Number of training iterations                                                                         | integer | 5           | True     | -                                                                                                                                                     |
| epsilon                     | Epsilon value for training                                                                            | number  | 0.5         | True     | -                                                                                                                                                     |
| per_device_train_batch_size | Per device batch size used for training                                                              | integer | 8           | True     | -                                                                                                                                                     |
| per_device_eval_batch_size  | Per device batch size used for evaluation                                                            | integer | 8           | True     | -                                                                                                                                                     |
| gradient_accumulation_steps | Number of steps to accumulate gradients before performing a backward pass                            | integer | 1           | True     | -                                                                                                                                                     |
| learning_rate               | Learning rate for training                                                                            | number  | 1e-6        | True     | -                                                                                                                                                     |
| logging_steps               | Number of steps between logging updates.                                                             | number  | 5           | True     | -                                                                                                                                                     |
| lr_scheduler_type           | The scheduler type to use for learning rate scheduling.                                              | string  | cosine      | True     | ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau']              |
| num_train_epochs            | Number of training epochs                                                                             | number  | 4.0         | True     | -                                                                                                                                                     |
| max_grad_norm               | Maximum gradient norm for gradient clipping                                                          | number  | 1.0         | True     | -                                                                                                                                                     |
| warmup_ratio                | Ratio of total training steps used for warmup                                                        | number  | 0.1         | True     | -                                                                                                                                                     |
| max_steps                   | If set to a positive number, this will override num_train_epochs and train for exactly this many steps. Set to -1 to disable (default). | integer | -1          | True     | -                                                                                                                                                     |
| eval_steps                  | Number of steps between evaluations                                                                  | integer | 1           | True     | -                                                                                                                                                     |
| optim                       | The optimizer to use.                                                                                | string  | adamw_torch | True     | ['adamw_torch', 'adamw_torch_fused', 'adafactor', 'ademamix', 'sgd', 'adagrad', 'rmsprop', 'galore_adamw', 'grokadamw', 'schedule_free_sgd']       |
| use_liger_kernel            | Whether to use the Liger kernel                                                                      | boolean | false       | True     | -                                                                                                                                                     |
| deepspeed_config            | Path to a custom DeepSpeed configuration file in JSON format                                         | uri_file| -           | False    | -                                                                                                                                                     |
| do_eval                     | Whether to run evaluation on the validation set or not. Will be set to True if eval_strategy is different from "no" | boolean | false       | True     | -                                                                                                                                                     |

### Generation Parameters

| Name                        | Description                                               | Type    | Default | Optional |
| --------------------------- | --------------------------------------------------------- | ------- | ------- | -------- |
| max_prompt_length           | Maximum length of the input prompt                       | integer | 512     | True     |
| num_generations             | Number of generations to produce                          | integer | 4       | True     |
| max_completion_length       | Maximum length of the completion                          | integer | 256     | True     |
| temperature                 | Temperature for sampling                                  | number  | 1.0     | True     |
| num_completions_to_print    | Number of completions to print during evaluation for inspection | integer | 2       | True     |
| top_p                       | Top-p value for nucleus sampling                          | number  | 1.0     | True     |

### Checkpoint and Dataset Parameters

| Name                  | Description                                         | Type    | Default | Optional |
| --------------------- | --------------------------------------------------- | ------- | ------- | -------- |
| save_steps            | Number of steps between saving checkpoints.        | integer | 100     | True     |
| save_total_limit      | Maximum number of checkpoints to keep.             | integer | 20      | True     |
| shuffle_dataset       | Whether to shuffle the dataset                      | boolean | true    | True     |

### VLLM Parameters

| Name                         | Description                               | Type    | Default | Optional |
| ---------------------------- | ----------------------------------------- | ------- | ------- | -------- |
| vllm_gpu_memory_utilization  | GPU memory utilization for VLLM          | number  | 0.3     | True     |
| vllm_tensor_parallel_size    | Tensor parallel size for VLLM            | integer | 1       | True     |

### Additional Training Parameters

| Name | Description                    | Type   | Default | Optional |
| ---- | ------------------------------ | ------ | ------- | -------- |
| beta | Beta parameter for training    | number | 0.04    | True     |

### Compute Parameters

| Name                 | Description                                                                                                                                                                                                                                                                            | Type   | Default    | Optional |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ---------- | -------- |
| compute_model_import | compute to be used for model_import eg. provide 'FT-Cluster' if your compute is named 'FT-Cluster'. Special characters like \ and ' are invalid in the parameter value. If compute cluster name is provided, instance_type field will be ignored and the respective cluster will be used | string | serverless | True     |
| compute_finetune     | compute to be used for finetune eg. provide 'FT-Cluster' if your compute is named 'FT-Cluster'. Special characters like \ and ' are invalid in the parameter value. If compute cluster name is provided, instance_type field will be ignored and the respective cluster will be used   | string | serverless | True     |

## Outputs

| Name                | Description                                                   | Type         |
| ------------------- | ------------------------------------------------------------- | ------------ |
| output_model_path   | Path to the output model folder containing the checkpoints   | uri_folder   |
| mlflow_model_folder | output folder containing _best_ finetuned model in mlflow format. | mlflow_model |