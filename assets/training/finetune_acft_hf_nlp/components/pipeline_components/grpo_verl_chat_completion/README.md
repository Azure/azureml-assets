## Verl Finetune Pipeline

### Name

verl_finetune_pipeline

### Version

0.0.2

### Type

pipeline

### Description

Pipeline component for fine-tuning models using Verl Package with Group Relative Policy Optimization (GRPO) and other reinforcement learning algorithms

## Inputs

### Infrastructure Parameters

| Name                            | Description                                                                                                                                                                                     | Type    | Default            | Optional |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------------------ | -------- |
| instance_type_model_import      | Instance type to be used for model_import component in case of serverless compute, eg. standard_d12_v2. The parameter compute_model_import must be set to 'serverless' for instance_type to be used | string  | Standard_d12_v2    | True     |
| instance_type_finetune          | Instance type to be used for finetune component in case of serverless compute, eg. standard_nc24rs_v3. The parameter compute_finetune must be set to 'serverless' for instance_type to be used     | string  | Standard_ND96isr_H100_v5 | True     |
| shm_size_finetune               | Shared memory size to be used for finetune component. It is useful while using Nebula (via DeepSpeed) which uses shared memory to save model and optimizer states.                            | string  | 5g                 | True     |
| num_nodes_finetune              | number of nodes to be used for finetuning (used for distributed training)                                                                                                                      | integer | 1                  | True     |
| number_of_gpu_to_use_finetuning | number of gpus to be used per node for finetuning, should be equal to number of gpu per node in the compute SKU used for finetune                                                              | integer | 1                  | True     |

### Model Import Parameters

| Name               | Description                                                                                                                                                                                                                                                                                                                                                                                                                   | Type         | Default | Optional |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------- | -------- |
| huggingface_id     | The string can be any valid Hugging Face id from the [Hugging Face models webpage](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads). Models from Hugging Face are subject to third party license terms available on the Hugging Face model details page. It is your responsibility to comply with the model's license terms. | string       | -       | True     |
| pytorch_model_path | Pytorch model asset path. Special characters like \ and ' are invalid in the parameter value.                                                                                                                                                                                                                                                                                                                                 | custom_model | -       | True     |
| mlflow_model_path  | MLflow model asset path. Special characters like \ and ' are invalid in the parameter value.                                                                                                                                                                                                                                                                                                                                  | mlflow_model | -       | True     |

### Engine and Data Parameters

| Name                        | Description                                   | Type     | Default | Optional |
| --------------------------- | --------------------------------------------- | -------- | ------- | -------- |
| ENGINE                      | Engine type (default: vllm)                   | string   | vllm    | True     |
| data_train_files            | Path to the training parquet file            | uri_file | -       | False    |
| data_val_files              | Path to the validation parquet file          | uri_file | -       | False    |
| data_train_batch_size       | Training batch size                           | integer  | 512     | True     |
| data_max_prompt_length      | Maximum prompt length                         | integer  | 1024    | True     |
| data_max_response_length    | Maximum response length                       | integer  | 2048    | True     |
| data_filter_overlong_prompts| Filter overlong prompts                       | boolean  | true    | True     |
| data_truncation             | Truncation strategy                           | string   | error   | True     |
| data_image_key              | Image key column                              | string   | images  | True     |

### Actor Model Parameters

| Name                                    | Description                                              | Type    | Default     | Optional |
| --------------------------------------- | -------------------------------------------------------- | ------- | ----------- | -------- |
| actor_optim_lr                          | Actor optimizer learning rate                            | number  | 3e-6        | True     |
| actor_model_use_remove_padding          | Use remove padding in model                              | boolean | true        | True     |
| actor_strategy                          | Actor training strategy (e.g., fsdp, fsdp2)             | string  | fsdp2       | True     |
| actor_fsdp_config_offload_policy        | FSDP config offload policy to reduce memory usage       | boolean | true        | True     |
| actor_ppo_mini_batch_size               | PPO mini batch size                                      | integer | 128         | True     |
| actor_ppo_micro_batch_size_per_gpu      | PPO micro batch size per GPU                             | integer | 10          | True     |
| actor_model_lora_rank                   | LoRA rank                                                | integer | 64          | True     |
| actor_model_lora_alpha                  | LoRA alpha                                               | integer | 32          | True     |
| actor_model_target_modules              | Target modules for LoRA                                  | string  | all-linear  | True     |
| actor_model_exclude_modules             | Exclude modules regex                                    | string  | .*visual.*  | True     |
| actor_use_kl_loss                       | Use KL loss                                              | boolean | true        | True     |
| actor_kl_loss_coef                      | KL loss coefficient                                      | number  | 0.01        | True     |
| actor_kl_loss_type                      | KL loss type                                             | string  | low_var_kl  | True     |
| actor_entropy_coeff                     | Entropy coefficient                                      | integer | 0           | True     |
| actor_model_enable_gradient_checkpointing | Enable gradient checkpointing                          | boolean | true        | True     |
| actor_fsdp_param_offload                | FSDP param offload                                       | boolean | false       | True     |
| actor_fsdp_optimizer_offload            | FSDP optimizer offload                                   | boolean | false       | True     |

### Rollout Parameters

| Name                                    | Description                                              | Type    | Default   | Optional |
| --------------------------------------- | -------------------------------------------------------- | ------- | --------- | -------- |
| rollout_log_prob_micro_batch_size_per_gpu | Rollout log prob micro batch size per GPU              | integer | 20        | True     |
| rollout_tensor_model_parallel_size      | Rollout tensor model parallel size                       | integer | 2         | True     |
| rollout_name                            | Rollout name (engine)                                    | string  | vllm      | True     |
| rollout_dtype                           | Rollout data type (e.g., float16, bfloat16, float32)   | string  | float16   | True     |
| rollout_disable_mm_preprocessor_cache   | Disable MM preprocessor cache                            | boolean | true      | True     |
| rollout_gpu_memory_utilization          | Rollout GPU memory utilization                           | number  | 0.6       | True     |
| rollout_enable_chunked_prefill          | Enable chunked prefill                                   | boolean | false     | True     |
| rollout_enforce_eager                   | Enforce eager execution                                  | boolean | false     | True     |
| rollout_free_cache_engine               | Free cache engine                                        | boolean | false     | True     |
| rollout_n                               | Rollout n                                                | integer | 5         | True     |

### Reference Model Parameters

| Name                                    | Description                                  | Type    | Default | Optional |
| --------------------------------------- | -------------------------------------------- | ------- | ------- | -------- |
| ref_log_prob_micro_batch_size_per_gpu   | Ref log prob micro batch size per GPU       | integer | 20      | True     |
| ref_fsdp_param_offload                  | Ref FSDP param offload                       | boolean | true    | True     |

### Algorithm Parameters

| Name                      | Description                                                      | Type    | Default | Optional |
| ------------------------- | ---------------------------------------------------------------- | ------- | ------- | -------- |
| algorithm_adv_estimator   | Advantage estimator algorithm (e.g., grpo, gae, reinforce_plus_plus, etc.) | string  | grpo    | True     |
| algorithm_use_kl_in_reward| Use KL in reward                                                 | boolean | false   | True     |

### Trainer Parameters

| Name                     | Description                   | Type    | Default                       | Optional |
| ------------------------ | ----------------------------- | ------- | ----------------------------- | -------- |
| trainer_critic_warmup    | Critic warmup                 | integer | 0                             | True     |
| trainer_n_gpus_per_node  | Number of GPUs per node       | integer | 8                             | True     |
| trainer_nnodes           | Number of nodes               | integer | 1                             | True     |
| trainer_save_freq        | Save frequency                | integer | 20                            | True     |
| trainer_test_freq        | Test frequency                | integer | 5                             | True     |
| trainer_total_epochs     | Total epochs                  | integer | 15                            | True     |
| total_training_steps     | Total number of training steps| integer | -                             | True     |

### Compute Parameters

| Name                 | Description                                                                                                                                                                                                                                                                            | Type   | Default    | Optional |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ---------- | -------- |
| compute_model_import | compute to be used for model_import eg. provide 'FT-Cluster' if your compute is named 'FT-Cluster'. Special characters like \ and ' are invalid in the parameter value. If compute cluster name is provided, instance_type field will be ignored and the respective cluster will be used | string | serverless | True     |
| compute_finetune     | compute to be used for finetune eg. provide 'FT-Cluster' if your compute is named 'FT-Cluster'. Special characters like \ and ' are invalid in the parameter value. If compute cluster name is provided, instance_type field will be ignored and the respective cluster will be used   | string | serverless | True     |

## Outputs

| Name         | Description                                         | Type       |
| ------------ | --------------------------------------------------- | ---------- |
| model_output | Directory containing the trained model artifacts   | uri_folder |
