## Text Generation Pipeline

### Name 

text_generation_pipeline

### Version 

0.0.17

### Type 

pipeline

### Description 

Pipeline component for text generation

## Inputs 

Compute parameters

| Name                            | Description                                                                                                                                                                                                     | Type    | Default            | Optional | Enum |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------------------ | -------- | ---- |
| instance_type_model_import      | Instance type to be used for model_import component in case of serverless compute, eg. standard_d12_v2. The parameter compute_model_import must be set to 'serverless' for instance_type to be used             | string  | Standard_d12_v2    | True     | NA   |
| instance_type_preprocess        | Instance type to be used for preprocess component in case of serverless compute, eg. standard_d12_v2. The parameter compute_preprocess must be set to 'serverless' for instance_type to be used                 | string  | Standard_d12_v2    | True     | NA   |
| instance_type_finetune          | Instance type to be used for finetune component in case of serverless compute, eg. standard_nc24rs_v3. The parameter compute_finetune must be set to 'serverless' for instance_type to be used                  | string  | Standard_nc24rs_v3 | True     | NA   |
| instance_type_model_evaluation  | Instance type to be used for model_evaluation components in case of serverless compute, eg. standard_nc24rs_v3. The parameter compute_model_evaluation must be set to 'serverless' for instance_type to be used | string  | Standard_nc24rs_v3 | True     | NA   |
| num_nodes_finetune              | number of nodes to be used for finetuning (used for distributed training)                                                                                                                                       | integer | 1                  | True     | NA   |
| number_of_gpu_to_use_finetuning | number of gpus to be used per node for finetuning, should be equal to number of gpu per node in the compute SKU used for finetune                                                                               | integer | 1                  | True     | NA   |



ModelSelector parameters

| Name           | Description                                                                                                                                                                                                                                                           | Type   | Default | Optional | Enum |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------- | -------- | ---- |
| huggingface_id | Input HuggingFace model id. Incase of continual finetuning provide proper id. Models from Hugging Face are subject to third party license terms available on the Hugging Face model details page. It is your responsibility to comply with the model's license terms. | string | -       | True     | NA   |



Continual-Finetuning model path

| Name               | Description                                                                                                    | Type         | Default | Optional | Enum |
| ------------------ | -------------------------------------------------------------------------------------------------------------- | ------------ | ------- | -------- | ---- |
| pytorch_model_path | Input folder path containing pytorch model for further finetuning. Proper model/huggingface id must be passed. | custom_model | -       | True     | NA   |
| mlflow_model_path  | Input folder path containing mlflow model for further finetuning. Proper model/huggingface id must be passed.  | mlflow_model | -       | True     | NA   |



Preprocessing parameters

| Name              | Description                                                                                                                                                                                                                                                                                                                                                                                                                     | Type    | Default | Optional | Enum              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------- | -------- | ----------------- |
| text_key          | key for text in an example. format your data keeping in mind that text is concatenated with ground_truth while finetuning in the form - text + groundtruth. for eg. "text"="knock knock\n", "ground_truth"="who's there"; will be treated as "knock knock\nwho's there"                                                                                                                                                         | string  | -       | False    | NA                |
| ground_truth_key  | key for ground_truth in an example. we take separate column for ground_truth to enable use cases like summarization, translation, question_answering, etc. which can be repurposed in form of text-generation where both text and ground_truth are needed. This separation is useful for calculating metrics. for eg. "text"="Summarize this dialog:\n{input_dialogue}\nSummary:\n", "ground_truth"="{summary of the dialogue}" | string  | -       | False    | NA                |
| batch_size        | Number of examples to batch before calling the tokenization function                                                                                                                                                                                                                                                                                                                                                            | integer | 1000    | True     | NA                |
| pad_to_max_length | If set to True, the returned sequences will be padded according to the model's padding side and padding index, up to their `max_seq_length`. If no `max_seq_length` is specified, the padding is done up to the model's max length.                                                                                                                                                                                             | string  | false   | True     | ['true', 'false'] |
| max_seq_length    | Default is -1 which means the padding is done up to the model's max length. Else will be padded to `max_seq_length`.                                                                                                                                                                                                                                                                                                            | integer | -1      | True     | NA                |



Dataset path Parameters

| Name                    | Description                       | Type     | Default | Optional | Enum |
| ----------------------- | --------------------------------- | -------- | ------- | -------- | ---- |
| train_file_path         | Enter the train file path         | uri_file | -       | True     | NA   |
| validation_file_path    | Enter the validation file path    | uri_file | -       | True     | NA   |
| test_file_path          | Enter the test file path          | uri_file | -       | True     | NA   |
| train_mltable_path      | Enter the train mltable path      | mltable  | -       | True     | NA   |
| validation_mltable_path | Enter the validation mltable path | mltable  | -       | True     | NA   |
| test_mltable_path       | Enter the test mltable path       | mltable  | -       | True     | NA   |



Finetuning parameters

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
| optim                       | Optimizer to be used while training                                                                                                                                                                                                                                                | string  | adamw_torch | True     | ['adamw_torch', 'adafactor']                                                       |
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



Model Evaluation parameters

| Name                     | Description                                                                                                      | Type     | Default | Optional | Enum |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- | -------- | ------- | -------- | ---- |
| evaluation_config        | Additional parameters for Computing Metrics. Special characters like \ and ' are invalid in the parameter value. | uri_file | -       | True     | NA   |
| evaluation_config_params | Additional parameters as JSON serielized string                                                                  | string   | -       | True     | NA   |



Compute parameters

| Name                     | Description                                                                                                                                                                                                                                                                                   | Type   | Default    | Optional | Enum |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ---------- | -------- | ---- |
| compute_model_import     | compute to be used for model_import eg. provide 'FT-Cluster' if your compute is named 'FT-Cluster'. Special characters like \ and ' are invalid in the parameter value. If compute cluster name is provided, instance_type field will be ignored and the respective cluster will be used      | string | serverless | True     | NA   |
| compute_preprocess       | compute to be used for preprocess eg. provide 'FT-Cluster' if your compute is named 'FT-Cluster'. Special characters like \ and ' are invalid in the parameter value. If compute cluster name is provided, instance_type field will be ignored and the respective cluster will be used        | string | serverless | True     | NA   |
| compute_finetune         | compute to be used for finetune eg. provide 'FT-Cluster' if your compute is named 'FT-Cluster'. Special characters like \ and ' are invalid in the parameter value. If compute cluster name is provided, instance_type field will be ignored and the respective cluster will be used          | string | serverless | True     | NA   |
| compute_model_evaluation | compute to be used for model_eavaluation eg. provide 'FT-Cluster' if your compute is named 'FT-Cluster'. Special characters like \ and ' are invalid in the parameter value. If compute cluster name is provided, instance_type field will be ignored and the respective cluster will be used | string | serverless | True     | NA   |

## Outputs 

| Name                 | Description                                              | Type         |
| -------------------- | -------------------------------------------------------- | ------------ |
| pytorch_model_folder | Output dir to save the finetune model and other metadata | uri_folder   |
| mlflow_model_folder  | Output dir to save the finetune model as mlflow model    | mlflow_model |
| evaluation_result    | Test Data Evaluation Results                             | uri_folder   |