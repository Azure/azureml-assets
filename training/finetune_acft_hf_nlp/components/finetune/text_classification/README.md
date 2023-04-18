# Text Classification Finetune Component
This component enables finetuning of Hugging Face pretrained models on custom or pre-available datasets. The component supports LoRA, Deepspeed and ONNXRuntime configurations for performance enhancement.

# 1. Inputs
1. _deepspeed_ (URI_FILE, optional)

    Input path to the deepspeed config file. This is a JSON file that can be used to configure optimizer, scheduler, batch size and other training related parameters. A default deepspeed config is used when _apply_deepspeed_ is set to `true`. Alternatively, you can pass your custom deepspeed config. Please follow the [deepspeed docs](https://www.deepspeed.ai/docs/config-json/) to create the custom config.

2. _preprocess_output_ (URI_FOLDER, required)

    Path to the output folder of Text Classification DataPreProcess component

3. _model_selector_output_ (URI_FOLDER, required)

    Path to the output directory of Text Classification Model Import component


# 2. Outputs
1. _pytorch_model_folder_ (URI_FOLDER)

    The folder containing _best_ model as defined by _metric_for_best_model_. Along with the best model, output folder contains checkpoints saved after every evaluation which is defined by the _evaluation_strategy_. Each checkpoint contains the model weight(s), config, tokenizer, optimzer, scheduler and random number states.

2. _mlflow_model_folder_ (MLFLOW_MODEL)

    The folder containing _best_ finetuned model in mlflow format.

# 3. Parameters
    
## 3.1. Lora Parameters
1. _apply_lora_ (string, optional)

    If "true" enables lora. The default is "false". The lora is `ONLY` supported for following model families -
    1. GPT2
    2. BERT
    3. ROBERTA
    4. DEBERTA
    5. DISTILBERT
    6. T5
    7. BART
    8. MBART
    9. CAMEMBERT

2. _merge_lora_weights_ (string, optional)

    If "true", the lora weights are merged with the base Hugging Face model. The default value is "true"

    NOTE When *model_evaluation component* is connected to *finetune component*, _merge_lora_weights_ **MUST** be set to "true" when _apply_lora_ is "true"

3. _lora_alpha_ (integer, optional)

    alpha attention parameter for lora. The default value is 128

4. _lora_r_ (integer, optional)

    The rank to be used with lora. The default value is 8

5. _lora_dropout_ (float, optional)

    lora dropout value. The default value is 0.0

## 3.2. Training Parameters

1. _num_train_epochs_ (int, optional)

    Number of epochs to run for finetune. The default value is 1

2. _max_steps_ (int, optional)

    If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`. In case of using a finite iterable dataset the training may stop before reaching the set number of steps when all data is exhausted. default value is -1

3. _learning_rate_ (float, optional)

    Start learning rate used for training. The default value is 2e-5

4. _per_device_train_batch_size_ (int, optional)

    Per gpu batch size used for training. The default value is 1. The effective training batch size is _per_device_train_batch_size_ * _num_gpus_ * _num_nodes_.

5. _per_device_eval_batch_size_ (int, optional)

    Per gpu batch size used for validation. The default value is 1. The effective validation batch size is _per_device_eval_batch_size_ * _num_gpus_ * _num_nodes_.

6. _auto_find_batch_size_ (string, optional)

    If set to "true", the train batch size will be automatically downscaled recursively till if finds a valid batch size that fits into memory. The default value is "false".

## 3.3. Deepspeed and ORT Parameters
1. _apply_ort_ (string, optional)

    If "true" apply Onnx RunTime optimization is applied during finetune. The default is "false".

2. _apply_deepspeed_ (string, optional)

    If "true" enables deepspeed. If no `deepspeed` is provided, the default config will be used else the user passed config will be used. The default is "false".

    Please note that to enable deepspeed, `apply_deepspeed` must be set to true, only passing the `deepspeed input` will not suffice


## 3.4. Optimizer and Scheduler Parameters

1. _optim_ (string, optional)

    Optimizer to be used while training. The default value is "adamw_hf". The other available optimizers are
    1. adamw_hf
    2. adamw_torch
    3. adafactor

2. _warmup_steps_ (int, optional)

    The number of steps for the learning rate scheduler warmup phase. The default value is 0

3. _weight_decay_ (float, optional)

    If not 0, the weight decay will be applied to all layers except all bias and LayerNorm weights in AdamW optimizer. The default value is 0

4. _adam_beta1_ (float, optional)

    The beta1 hyperparameter for the AdamW optimizer. The default value is 0.9

5. _adam_beta2_ (float, optional)

    The beta2 hyperparameter for the AdamW optimizer. The default value is 0.999

6. _adam_epsilon_ (float, optional)

    The epsilon hyperparameter for the AdamW optimizer. The default value is 1e-8

7. _gradient_accumulation_steps_ (int, optional)

    Number of updates steps to accumulate the gradients for, before performing a backward/update pass. The default value is 1

8. _lr_scheduler_type_ (string, optional)

    The learning rate scheduler to use. The default value is `linear`. The other available schedulers are
    1. linear
    2. cosine
    3. cosine_with_restarts
    4. polynomial
    5. constant
    6. constant_with_warmup

## 3.5. Misc Parameters

1. _precision_ (int, optional)

    Apply mixed precision training. This can reduce memory footprint by performing operations in half-precision. The supported precision values are 16 and 32. The default value is 32

2. _seed_ (int, optional)

    Random seed that will be set at the beginning of training. The default value is 42

3. _dataloader_num_workers_ (int, optional)

    Number of subprocesses to use for data loading. The default value is 0 which means that the data will be loaded in the main process.

4. _ignore_mismatched_sizes_ (string, optional)

    Not setting this flag will raise an error if some of the weights from the checkpoint do not have the same size as the weights of the model. The default value is `true`.

5. _max_grad_norm_ (float, optional)

    Maximum gradient norm (for gradient clipping). The default value is 1.0.

6. _evaluation_strategy_ (string, optional)

    The evaluation strategy to adopt during training. If set to "steps", either the `evaluation_steps_interval` or `eval_steps` needs to be specified, which helps to determine the step at which the model evaluation needs to be computed else evaluation happens at end of each epoch. The default value is "epoch". At the end of every evaluation, a model checkpoint gets saved.

7. _evaluation_steps_interval_ (float, optional)

    The evaluation steps in fraction of an epoch steps to adopt during training. Overwrites _eval_steps_ if not 0. The default value is 0

8. _eval_steps_ (int, optional)

    Number of update steps between two model evaluations if evaluation_strategy='steps'. The default value is 500

9. _logging_strategy_ (string, optional)

    The logging strategy to adopt during training. If set to "steps", the `logging_steps` will decide the frequency of logging else logging happens at the end of epoch. The default value is "epoch".

10. _logging_steps_ (int, optional)

    Number of update steps between two logs if logging_strategy='steps'. The default value is 500

11. _save_total_limit_ (int, optional)

    If a value is passed, will limit the total amount of checkpoints. The value of -1 saves all the checkpoints, otherwise if the number of checkpoints exceed the _save_total_limit_, the older checkpoints gets deleted. The default value is -1

12. _apply_early_stopping_ (string, optional)

    If set to "true", early stopping is enabled. The default value is "false"

13. _early_stopping_patience_ (int, optional)

    Stop training when the metric specified through _metric_for_best_model_ worsens for _early_stopping_patience_ evaluation calls.This value is only valid if _apply_early_stopping_ is set to true. The default value is 1

14. _early_stopping_threshold_ (float, optional)

    Denotes how much the specified metric must improve to satisfy early stopping conditions. This value is only valid if _apply_early_stopping_ is set to true. The default value is 0.0

15. _metric_for_best_model_ (string, optional)

    Metric to be used to compare two different models. The possible metrics are loss, f1_macro, accuracy_macro, precision_macro, recall_macro and Matthews Correlation Coefficient (MCC). The default metric used is loss


## 3.6. Continual Finetuning

1. _resume_from_checkpoint_ (string, optional)

    If set to "true", resumes the training from last saved checkpoint. Along with loading the saved weights, saved optimizer, scheduler and random states will be loaded if exist. The default value is "false"
