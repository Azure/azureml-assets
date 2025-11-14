## Verl Trainer Component

### Name

verl_trainer_component

### Version

0.0.1

### Type

command

### Description

Component for Verl Finetuning. Supports reinforcement learning algorithms including GRPO (Group Relative Policy Optimization) with PyTorch distributed training using FSDP.

## Inputs

### Engine and Data Parameters

| Name                        | Description                                   | Type     | Default | Optional | Enum |
| --------------------------- | --------------------------------------------- | -------- | ------- | -------- | ---- |
| ENGINE                      | Engine type (default: vllm)                   | string   | vllm    | True     | NA   |
| data_train_files            | Path to the training parquet file            | uri_file | -       | False    | NA   |
| data_val_files              | Path to the validation parquet file          | uri_file | -       | False    | NA   |
| data_train_batch_size       | Training batch size                           | integer  | 512     | True     | NA   |
| data_max_prompt_length      | Maximum prompt length                         | integer  | 1024    | True     | NA   |
| data_max_response_length    | Maximum response length                       | integer  | 2048    | True     | NA   |
| data_filter_overlong_prompts| Filter overlong prompts                       | boolean  | true    | True     | NA   |
| data_truncation             | Truncation strategy                           | string   | error   | True     | NA   |
| data_image_key              | Image key column                              | string   | images  | True     | NA   |

### Model Parameters

| Name                                    | Description                                                                     | Type       | Default     | Optional | Enum |
| --------------------------------------- | ------------------------------------------------------------------------------- | ---------- | ----------- | -------- | ---- |
| actor_model_path                        | Output folder of model import component containing model artifacts and a metadata file. | uri_folder | -           | False    | NA   |

### Actor Model Training Parameters

| Name                                    | Description                                              | Type    | Default     | Optional | Enum              |
| --------------------------------------- | -------------------------------------------------------- | ------- | ----------- | -------- | ----------------- |
| actor_optim_lr                          | Actor optimizer learning rate                            | number  | 3e-6        | True     | NA                |
| actor_model_use_remove_padding          | Use remove padding in model                              | boolean | true        | True     | NA                |
| actor_strategy                          | Actor training strategy. Valid values: fsdp (Fully Sharded Data Parallel), fsdp2 (FSDP version 2) | string  | fsdp2       | True     | ['fsdp', 'fsdp2'] |
| actor_fsdp_config_offload_policy        | FSDP config offload policy to reduce memory usage       | boolean | true        | True     | NA                |
| actor_ppo_mini_batch_size               | PPO mini batch size                                      | integer | 128         | True     | NA                |
| actor_ppo_micro_batch_size_per_gpu      | PPO micro batch size per GPU                             | integer | 10          | True     | NA                |
| actor_model_lora_rank                   | LoRA rank                                                | integer | 64          | True     | NA                |
| actor_model_lora_alpha                  | LoRA alpha                                               | integer | 32          | True     | NA                |
| actor_model_target_modules              | Target modules for LoRA                                  | string  | all-linear  | True     | NA                |
| actor_model_exclude_modules             | Exclude modules regex                                    | string  | .*visual.*  | True     | NA                |
| actor_use_kl_loss                       | Use KL loss                                              | boolean | true        | True     | NA                |
| actor_kl_loss_coef                      | KL loss coefficient                                      | number  | 0.01        | True     | NA                |
| actor_kl_loss_type                      | KL loss type                                             | string  | low_var_kl  | True     | NA                |
| actor_entropy_coeff                     | Entropy coefficient                                      | integer | 0           | True     | NA                |
| actor_model_enable_gradient_checkpointing | Enable gradient checkpointing                          | boolean | true        | True     | NA                |
| actor_fsdp_param_offload                | FSDP param offload                                       | boolean | false       | True     | NA                |
| actor_fsdp_optimizer_offload            | FSDP optimizer offload                                   | boolean | false       | True     | NA                |

### Rollout Parameters

| Name                                    | Description                                                                                                                                       | Type    | Default   | Optional | Enum                                    |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------- | -------- | --------------------------------------- |
| rollout_log_prob_micro_batch_size_per_gpu | Rollout log prob micro batch size per GPU                                                                                                       | integer | 20        | True     | NA                                      |
| rollout_tensor_model_parallel_size      | Rollout tensor model parallel size                                                                                                                | integer | 2         | True     | NA                                      |
| rollout_name                            | Rollout name (engine)                                                                                                                            | string  | vllm      | True     | NA                                      |
| rollout_dtype                           | Rollout data type for model inference. Valid values: float16 (half precision), bfloat16 (brain floating point), float32 (full precision)       | string  | float16   | True     | ['float16', 'bfloat16', 'float32']      |
| rollout_disable_mm_preprocessor_cache   | Disable MM preprocessor cache                                                                                                                    | boolean | true      | True     | NA                                      |
| rollout_gpu_memory_utilization          | Rollout GPU memory utilization                                                                                                                   | number  | 0.6       | True     | NA                                      |
| rollout_enable_chunked_prefill          | Enable chunked prefill                                                                                                                           | boolean | false     | True     | NA                                      |
| rollout_enforce_eager                   | Enforce eager execution                                                                                                                          | boolean | false     | True     | NA                                      |
| rollout_free_cache_engine               | Free cache engine                                                                                                                                | boolean | false     | True     | NA                                      |
| rollout_n                               | Rollout n                                                                                                                                        | integer | 5         | True     | NA                                      |

### Reference Model Parameters

| Name                                    | Description                                  | Type    | Default | Optional | Enum |
| --------------------------------------- | -------------------------------------------- | ------- | ------- | -------- | ---- |
| ref_log_prob_micro_batch_size_per_gpu   | Ref log prob micro batch size per GPU       | integer | 20      | True     | NA   |
| ref_fsdp_param_offload                  | Ref FSDP param offload                       | boolean | true    | True     | NA   |

### Algorithm Parameters

| Name                      | Description                                                                                                                                                                                                                                                                                                                                                                                                      | Type    | Default | Optional | Enum                                                                                                                    |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------- | -------- | ----------------------------------------------------------------------------------------------------------------------- |
| algorithm_adv_estimator   | Advantage estimator algorithm. Valid values: gae (Generalized Advantage Estimation), grpo (Group Relative Policy Optimization), reinforce_plus_plus (REINFORCE++), reinforce_plus_plus_baseline (REINFORCE++ with baseline), remax (ReMax algorithm), rloo (Reinforcement Learning with Leave-One-Out), opo (Outcome Policy Optimization), grpo_passk (GRPO for Pass@k evaluation), gpg (Generalized Policy Gradient) | string  | grpo    | True     | ['gae', 'grpo', 'reinforce_plus_plus', 'reinforce_plus_plus_baseline', 'remax', 'rloo', 'opo', 'grpo_passk', 'gpg'] |
| algorithm_use_kl_in_reward| Use KL in reward                                                                                                                                                                                                                                                                                                                                                                                                 | boolean | false   | True     | NA                                                                                                                      |

### Trainer Parameters

| Name                     | Description                   | Type    | Default                       | Optional | Enum |
| ------------------------ | ----------------------------- | ------- | ----------------------------- | -------- | ---- |
| trainer_critic_warmup    | Critic warmup                 | integer | 0                             | True     | NA   |
| trainer_n_gpus_per_node  | Number of GPUs per node       | integer | 8                             | True     | NA   |
| trainer_nnodes           | Number of nodes               | integer | 1                             | True     | NA   |
| trainer_save_freq        | Save frequency                | integer | 20                            | True     | NA   |
| trainer_test_freq        | Test frequency                | integer | 5                             | True     | NA   |
| trainer_total_epochs     | Total epochs                  | integer | 15                            | True     | NA   |
| total_training_steps     | Total number of training steps| integer | -                             | True     | NA   |

## Outputs

| Name         | Description                                         | Type       |
| ------------ | --------------------------------------------------- | ---------- |
| model_output | Directory containing the trained model artifacts   | uri_folder |
