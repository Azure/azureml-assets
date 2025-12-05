## Component Draft Model Trainer

### Name

component_draft_model_trainer

### Version

0.0.1

### Type

command

### Description

Component to train draft model for Speculative Decoding. This component trains Eagle3 draft models that can be used with target models to accelerate inference through speculative decoding techniques.

## Inputs

Dataset Parameters

| Name                      | Description                                                       | Type     | Default | Optional | Enum |
| ------------------------- | ----------------------------------------------------------------- | -------- | ------- | -------- | ---- |
| dataset_train_split       | Path to the training dataset in JSONL format.                     | uri_file | -       | True     | NA   |
| dataset_validation_split  | Path to the validation dataset in JSONL format.                   | uri_file | -       | True     | NA   |
| target_model_path         | Path to the target model for speculative decoding.                | uri_folder | -     | False    | NA   |
| draft_model_config        | Path to draft model configuration JSON file. If not provided, will be auto-generated from target model. | uri_file | - | True | NA |

Training Parameters

| Name                      | Description                                                       | Type    | Default | Optional | Enum |
| ------------------------- | ----------------------------------------------------------------- | ------- | ------- | -------- | ---- |
| num_epochs                | Number of training epochs for draft model.                        | integer | 2       | True     | NA   |
| batch_size                | Batch size for draft model training.                              | integer | 2       | True     | NA   |
| learning_rate             | Learning rate for training.                                       | number  | 0.0001  | True     | NA   |
| max_length                | Maximum sequence length for draft model training.                 | integer | 2048    | True     | NA   |
| warmup_ratio              | Warmup ratio for learning rate scheduler.                         | number  | 0.015   | True     | NA   |
| max_grad_norm             | Maximum gradient norm for gradient clipping.                      | number  | 0.5     | True     | NA   |
| ttt_length                | The length for Training-Time Test (TTT).                          | integer | 7       | True     | NA   |
| chat_template             | Chat template to use for formatting conversations. Supported templates - qwen, llama3, gpt-oss, deepseek-r1-distill. | string | llama3 | True | ['llama3', 'qwen', 'gpt-oss', 'deepseek-r1-distill'] |
| attention_backend         | Attention implementation backend to use.                          | string  | flex_attention | True | ['flex_attention', 'sdpa'] |
| seed                      | Random seed for reproducibility.                                  | integer | 0       | True     | NA   |
| total_steps               | Total training steps. Set to -1 or 0 to auto-calculate based on num_epochs and dataset size. If set to a positive value, will override num_epochs. | integer | -1 | True | NA |

Parallelism and Batch Configuration

| Name                       | Description                                                      | Type    | Default | Optional | Enum |
| -------------------------- | ---------------------------------------------------------------- | ------- | ------- | -------- | ---- |
| tp_size                    | Tensor parallelism size                                          | integer | 1       | True     | NA   |
| dp_size                    | Data parallelism size                                            | integer | 1       | True     | NA   |
| draft_global_batch_size    | Global batch size for draft model training                       | integer | 8       | True     | NA   |
| draft_micro_batch_size     | Micro batch size for draft model                                 | integer | 1       | True     | NA   |
| draft_accumulation_steps   | Gradient accumulation steps for draft model                      | integer | 1       | True     | NA   |

Logging and Checkpointing Parameters

| Name                       | Description                                                      | Type    | Default | Optional | Enum |
| -------------------------- | ---------------------------------------------------------------- | ------- | ------- | -------- | ---- |
| log_steps                  | Log training metrics every N steps                               | integer | 50      | True     | NA   |
| eval_interval              | Evaluation interval in epochs                                    | integer | 1       | True     | NA   |
| save_interval              | Checkpoint save interval in epochs                               | integer | 1       | True     | NA   |
| resume                     | Whether to resume training from the last checkpoint              | boolean | false   | True     | NA   |
| resume_from_checkpoint     | Path to a checkpoint directory to resume training from. Used when resume is true and no checkpoints exist in output folder, or when resume is false to initialize from a pretrained draft model checkpoint. | uri_folder | - | True | NA |

Advanced Parameters

| Name                      | Description                                                       | Type    | Default | Optional | Enum |
| ------------------------- | ----------------------------------------------------------------- | ------- | ------- | -------- | ---- |
| dist_timeout              | Timeout for collective communication in minutes                   | integer | 20      | True     | NA   |
| build_dataset_num_proc    | Number of processes to use for building the dataset. Recommended to set same as number of CPU cores available. If this is too high, one may loose performance due to context switching. | integer | 96 | True | NA |

## Outputs

| Name              | Description                                                           | Type       |
| ----------------- | --------------------------------------------------------------------- | ---------- |
| output_model_path | Output folder containing trained Eagle3 draft model checkpoints       | uri_folder |
