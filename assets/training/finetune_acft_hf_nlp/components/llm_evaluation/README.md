## Component Model Evaluation

### Name

component_model_evaluation

### Version

0.0.1

### Type

command

### Description

Component for model response evaluation across multiple models. This component enables comprehensive evaluation of fine-tuned models and checkpoints by comparing outputs across different training runs, checkpoints, and base models.

## Inputs

Model Checkpoint Parameters

| Name                    | Description                                                                                                                               | Type       | Default                                    | Optional | Enum |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------ | -------- | ---- |
| checkpoint_base_path_1  | Base path containing all model checkpoints or LoRA adapters (optional if using only hf_model_id)                                         | uri_folder | -                                          | True     | NA   |
| checkpoint_base_path_2  | Second base path containing model checkpoints or LoRA adapters (optional, for comparing models from different training runs)             | uri_folder | -                                          | True     | NA   |
| base_path_1_label       | Label to use as prefix in metrics for checkpoint_base_path_1 (e.g., 'experiment_a'). Defaults to 'base_path_1'                           | string     | base_path_1                                | True     | NA   |
| base_path_2_label       | Label to use as prefix in metrics for checkpoint_base_path_2 (e.g., 'experiment_b'). Defaults to 'base_path_2'                           | string     | base_path_2                                | True     | NA   |
| evaluate_base_model     | If true, also evaluate the base model after evaluating model checkpoints                                                                 | boolean    | false                                      | True     | NA   |
| explore_pattern_1       | Pattern to explore for checkpoint paths (e.g., global_step_{checkpoint}/actor/huggingface/). Only used with checkpoint_base_path_1       | string     | global_step_{checkpoint}/actor/huggingface/ | True     | NA   |
| explore_pattern_2       | Pattern to explore for checkpoint paths in checkpoint_base_path_2 (e.g., global_step_{checkpoint}/actor/huggingface/). Only used with checkpoint_base_path_2 | string     | global_step_{checkpoint}/actor/huggingface/ | True     | NA   |
| checkpoint_values_1     | Comma-separated list of model checkpoint values to evaluate (e.g., '100,129,20'). Optional if using only hf_model_id                     | string     | -                                          | True     | NA   |
| checkpoint_values_2     | Comma-separated list of model checkpoint values to evaluate from checkpoint_base_path_2 (e.g., '100,129,20'). Only used with checkpoint_base_path_2 | string     | -                                          | True     | NA   |
| use_lora_adapters_1     | If true, model checkpoints from checkpoint_base_path_1 will be treated as LoRA adapters to be loaded with base model. Base model must be specified via base_model_path or hf_model_id | boolean    | false                                      | True     | NA   |
| use_lora_adapters_2     | If true, model checkpoints from checkpoint_base_path_2 will be treated as LoRA adapters to be loaded with base model. Base model must be specified via base_model_path or hf_model_id | boolean    | false                                      | True     | NA   |

Base Model Parameters

| Name             | Description                                                                                                                                                          | Type       | Default | Optional | Enum |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------- | -------- | ---- |
| base_model_path  | Local base model path (used when use_lora_adapters is true). Mutually exclusive with hf_model_id                                                                    | uri_folder | -       | True     | NA   |
| hf_model_id      | Hugging Face model ID (e.g., 'meta-llama/Llama-2-7b-hf'). Can be used alone for direct evaluation or with use_lora_adapters for base model. Mutually exclusive with base_model_path when use_lora_adapters is true | string     | -       | True     | NA   |

Evaluation Data Parameters

| Name            | Description                                 | Type     | Default | Optional | Enum |
| --------------- | ------------------------------------------- | -------- | ------- | -------- | ---- |
| validation_file | Path to validation JSONL file for evaluation | uri_file | -       | False    | NA   |

Inference Parameters

| Name                 | Description                                                                                  | Type    | Default  | Optional | Enum                              |
| -------------------- | -------------------------------------------------------------------------------------------- | ------- | -------- | -------- | --------------------------------- |
| max_prompt_length    | Maximum length for input prompts                                                             | integer | 2048     | True     | NA                                |
| max_response_length  | Maximum length for model responses                                                           | integer | 1024     | True     | NA                                |
| batch_size           | Batch size for inference                                                                     | integer | 16       | True     | NA                                |
| temperature          | Temperature parameter for sampling during generation                                         | number  | 0.7      | True     | NA                                |
| top_p                | Top-p (nucleus) sampling parameter for generation                                            | number  | 0.9      | True     | NA                                |
| tensor_parallel_size | Number of GPUs to use for tensor parallelism                                                 | integer | 1        | True     | NA                                |
| gpu_memory_utilization | Fraction of GPU memory to use for model inference                                          | number  | 0.8      | True     | NA                                |
| dtype                | Data type to use for model weights                                                           | string  | bfloat16 | True     | ['float16', 'bfloat16', 'float32'] |
| extraction_method    | Method for extracting responses from model outputs                                           | string  | strict   | True     | ['strict', 'flexible']            |
| n_gpus_per_node      | Number of GPUs available per node                                                            | integer | 1        | True     | NA                                |
| number_of_trials     | Number of evaluation trials per checkpoint                                                   | integer | 1        | True     | NA                                |

## Outputs

| Name                 | Description                                                                 | Type       |
| -------------------- | --------------------------------------------------------------------------- | ---------- |
| evaluation_results   | Directory containing all checkpoint evaluation results                      | uri_folder |
| intermediate_folder  | Directory containing preprocessed checkpoints (with corrections applied)    | uri_folder |
