## Component RL Trainer

### Name

rft_trainer

### Version

0.0.1

### Type

command

### Description

Component for Multi-Strategy Reinforcement Learning Training of Large Language Models. Supports GRPO (Group Relative Policy Optimization) and REINFORCE++ algorithms with distributed training capabilities.

## Inputs

### Algorithm Configuration

| Name                        | Description                                                                                                                                                               | Type    | Default              | Optional | Enum                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | -------------------- | -------- | --------------------------------------- |
| algorithm_adv_estimator     | Advantage estimator algorithm. Valid values: grpo (Group Relative Policy Optimization), reinforce_plus_plus (REINFORCE++ with global advantage normalization)           | string  | grpo                 | False    | ['grpo', 'reinforce_plus_plus']        |
| algorithm_use_kl_in_reward  | Add KL divergence penalty to reward signal (classic PPO approach with adaptive KL). Set false for GRPO which uses kl_loss in actor loss instead                          | boolean | false                | True     | NA                                      |

### Trainer Configuration

| Name                      | Description                                                                                                                                      | Type    | Default | Optional | Enum |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------- | ------- | -------- | ---- |
| trainer_n_gpus_per_node   | Number of GPUs per node                                                                                                                          | integer | 8       | True     | NA   |
| trainer_nnodes            | Number of nodes                                                                                                                                  | integer | 1       | True     | NA   |
| trainer_save_freq         | Checkpoint save frequency in epochs. The trainer saves model state (HuggingFace-compatible model) every N epochs                                | integer | 20      | True     | NA   |
| trainer_test_freq         | Validation frequency in epochs. For example, test_freq=5 means validation runs at epochs 5, 10, 15, etc                                          | integer | 5       | True     | NA   |
| trainer_val_before_train  | Run validation before training starts. When enabled, the trainer evaluates the model on the validation set before any training occurs            | boolean | true    | True     | NA   |
| trainer_total_epochs      | Total epochs                                                                                                                                     | integer | 7       | True     | NA   |
| total_training_steps      | Total number of training steps                                                                                                                   | integer | -       | True     | NA   |

### Data Configuration

| Name                           | Description                                                                                                                                                  | Type     | Default | Optional | Enum                              |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | ------- | -------- | --------------------------------- |
| data_train_files               | Path to the training parquet or jsonl file                                                                                                                   | uri_file | -       | False    | NA                                |
| data_val_files                 | Path to the validation parquet or jsonl file                                                                                                                 | uri_file | -       | False    | NA                                |
| data_train_batch_size          | Training batch size                                                                                                                                          | integer  | 512     | True     | NA                                |
| data_max_prompt_length         | Maximum prompt length                                                                                                                                        | integer  | 1024    | True     | NA                                |
| data_max_response_length       | Maximum response length                                                                                                                                      | integer  | 2048    | True     | NA                                |
| data_filter_overlong_prompts   | Filter overlong prompts                                                                                                                                      | boolean  | true    | True     | NA                                |
| data_truncation                | Truncation strategy. Options: 'error', 'left', 'right', 'middle'                                                                                             | string   | error   | False    | ['error', 'left', 'right', 'middle'] |

### Actor Model Configuration

| Name                                      | Description                                                                                                                                          | Type       | Default     | Optional | Enum                 |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------- | -------- | -------------------- |
| actor_model_path                          | Output folder of model import component containing model artifacts                                                                                   | uri_folder | -           | False    | NA                   |
| actor_optim_lr                            | Actor optimizer learning rate                                                                                                                        | number     | 3e-6        | True     | NA                   |
| actor_model_use_remove_padding            | Enables sequence packing optimization, which removes padding tokens from inputs during training to save computation                                  | boolean    | true        | True     | NA                   |
| actor_strategy                            | Actor training strategy. Valid values: fsdp (Fully Sharded Data Parallel v1), fsdp2 (Fully Sharded Data Parallel v2)                                | string     | fsdp2       | True     | ['fsdp', 'fsdp2']    |
| actor_fsdp_config_offload_policy          | FSDP config offload policy (params/optimizers/gradients) to reduce memory usage                                                                     | boolean    | true        | True     | NA                   |
| actor_rft_mini_batch_size                 | The global batch size for splitting sampled data into multiple sub-batches during training                                                          | integer    | 128         | True     | NA                   |
| actor_rft_micro_batch_size_per_gpu        | The per-GPU batch size for each forward and backward pass during training                                                                            | integer    | 10          | True     | NA                   |
| actor_model_lora_rank                     | LoRA rank to control the capacity of the low-rank adaptation. Higher values increase model adaptability but also computational cost and memory usage | integer    | 64          | True     | NA                   |
| actor_model_lora_alpha                    | LoRA alpha to scale the LoRA updates. Higher values increase the influence of the LoRA layers on the overall model output                           | integer    | 32          | True     | NA                   |
| actor_model_target_modules                | Target modules for LoRA. Options: 'all-linear', '[q_proj,k_proj,v_proj,o_proj]', '[q_proj,v_proj]', '.*_proj' (Regex pattern)                      | string     | all-linear  | True     | NA                   |
| actor_use_kl_loss                         | If True, KL penalty is added to actor loss which is computed during training, else KL penalty is added to reward during rollout                      | boolean    | true        | True     | NA                   |
| actor_kl_loss_coef                        | Controls the strength of the KL penalty. Higher values encourage the model to stay closer to the reference policy                                   | number     | 0.01        | True     | NA                   |
| actor_kl_loss_type                        | Types of KL divergence loss calculation. Valid values: 'low_var_kl', 'kl', 'abs', 'mse'                                                             | string     | low_var_kl  | True     | ['low_var_kl', 'kl', 'abs', 'mse'] |
| actor_entropy_coeff                       | Entropy regularization coefficient in loss. Default 0.0 (disabled) is recommended for most cases including GRPO and math/reasoning tasks            | number     | 0.0         | True     | NA                   |
| actor_model_enable_gradient_checkpointing | Trades off compute for memory by saving only a subset of activations during the forward pass and recomputing them during the backward pass          | boolean    | true        | True     | NA                   |
| actor_fsdp_param_offload                  | In FSDP strategy, offload model parameters to CPU memory when not in use to reduce GPU memory consumption                                            | boolean    | false       | True     | NA                   |
| actor_fsdp_optimizer_offload              | In FSDP strategy, we offload optimizer state to CPU                                                                                                  | boolean    | false       | True     | NA                   |

### Rollout Configuration

| Name                                | Description                                                                                                                                         | Type    | Default   | Optional | Enum                                 |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------- | -------- | ------------------------------------ |
| rollout_log_prob_micro_batch_size_per_gpu | Rollout log prob micro batch size per GPU                                                                                                      | integer | 20        | True     | NA                                   |
| rollout_tensor_model_parallel_size  | Rollout tensor model parallel size                                                                                                                  | integer | 2         | True     | NA                                   |
| rollout_dtype                       | Rollout data type for model inference. Valid values: float16, bfloat16, float32, auto                                                              | string  | bfloat16  | True     | ['float16', 'bfloat16', 'float32', 'auto'] |
| rollout_gpu_memory_utilization      | Rollout GPU memory utilization                                                                                                                      | number  | 0.6       | True     | NA                                   |
| rollout_enable_chunked_prefill      | Enable chunked prefill. A feature for rollout in vLLM which allows to chunk large prefills into smaller chunks                                     | boolean | false     | True     | NA                                   |
| rollout_enforce_eager               | Disable CUDA graphs and use eager execution in vLLM. Set to true for NPU/Ascend/non-NVIDIA hardware or debugging                                   | boolean | false     | True     | NA                                   |
| rollout_free_cache_engine           | Free vLLM KV cache and offload model to CPU between rollout steps                                                                                  | boolean | false     | True     | NA                                   |
| rollout_n                           | Number of responses to sample per prompt during rollout. Critical for GRPO (requires n > 1)                                                        | integer | 5         | True     | NA                                   |
| rollout_temperature                 | Sampling temperature for rollout generation. Controls randomness during generation. Range: [0.1, 2.0]                                              | number  | 1.0       | True     | NA                                   |
| rollout_top_p                       | Nucleus sampling threshold. Samples from smallest set of tokens with cumulative probability > top_p. Range: [0.0, 1.0]                             | number  | 1.0       | True     | NA                                   |
| rollout_top_k                       | Top-k sampling limit. Use -1 (disabled) for vLLM rollout, 0 for HuggingFace rollout                                                                | integer | -1        | True     | NA                                   |
| rollout_do_sample                   | Enable sampling during generation. Use true for RL training (stochastic sampling), false for greedy decoding                                        | boolean | true      | True     | NA                                   |
| rollout_val_temperature             | Temperature for validation/evaluation generation. Lower than training temperature for more deterministic and reproducible evaluation                | number  | 0.7       | True     | NA                                   |
| rollout_val_top_p                   | Top-p for validation/evaluation. Lower than training for more focused sampling during evaluation                                                    | number  | 0.7       | True     | NA                                   |
| rollout_val_top_k                   | Top-k for validation. Use -1 (disabled) for vLLM                                                                                                    | integer | -1        | True     | NA                                   |

### Reference Policy Configuration

| Name                                  | Description                                                                                                                                        | Type    | Default | Optional | Enum |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------- | -------- | ---- |
| ref_log_prob_micro_batch_size_per_gpu | Per-GPU micro batch size for computing reference policy log probabilities. Used when use_kl_loss or use_kl_in_reward is enabled                   | integer | 20      | True     | NA   |
| ref_fsdp_param_offload                | Offload reference policy model parameters to CPU to save GPU memory. Strongly recommended to enable (true) as reference model is read-only        | boolean | true    | True     | NA   |

### Miscellaneous

| Name                     | Description                                                                                                                          | Type   | Default | Optional | Enum |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | ------ | ------- | -------- | ---- |
| pypi_packages_override   | Comma-separated list of PyPI packages to override before starting the run (e.g., transformers==4.30.0,torch==2.3.1)                 | string | -       | True     | NA   |

## Outputs

| Name                  | Description                                                  | Type       |
| --------------------- | ------------------------------------------------------------ | ---------- |
| intermediate_folder   | Intermediate directory for RL training checkpoints           | uri_folder |
| model_output          | Directory containing the trained model artifacts             | uri_folder |
