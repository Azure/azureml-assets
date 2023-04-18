# Summarization Model Import Component
The component copies the input model folder to the component output directory when the model is passed as an input to the `pytorch_model_path` or `mlflow_model_path` nodes. If `huggingface_id `is selected, the model is downloaded from Hugging Face CDN.


# 1. Inputs
1. _pytorch_model_path_ (custom_model, optional)

    Pytorch model as an input. This input model folder is expected to contain model, config and tokenizer files and optionally optimizer, scheduler and the random states. The files are expected to be in the [Hugging Face format](https://huggingface.co/bert-base-uncased/tree/main) and only **PyTorch** models are supported. Additionally, the input folder **MUST** contain the file `finetune_args.json` with *model_name_or_path* as one of the keys of the dictionary. This file is already created if you are using an already finetuned model from Azureml

    If you want to resume from previous training state, set *resume_from_checkpoint* flag to True in finetune component

2. _mlflow_model_path_ (mlflow_model, optional)

    MLflow model as an input. This input folder is expected to contain model, config and tokenizer files in a specific format as explained below. You could use the Model import pipeline to create a model of your own or refer to any of models in the Model Catalogue page if you want to manually create one. The MLflow output of a finetune model will be in correct format and no modification is needed.

    - All the configuration files should be stored in _data/config_ folder
    - All the model files should be stored in _data/model_ folder
    - All the tokenizer files should be kept in _data/tokenizer_ folder
    - **`MLmodel`** is a yaml file and this should contain _model_name_or_path_ information.

    > Currently _resume_from_checkpoint_ is **NOT** fully enabled with _mlflow_model_path_. Only the saved model weights can be reloaded but not the optimizer, scheduler and random states

**NOTE** The _pytorch_model_path_ take priority over _mlflow_model_path_, in case both inputs are passed


# 2. Outputs
1. _output_dir_ (URI_FOLDER):

    Path to output directory which contains the component metadata and the copied model data, saved under either _model_id_ or _huggingface_id_, when model is passed through input nodes or _model_id_. In cases, where _huggingface_id_ is passed, only the component metadata is present in the output folder.


# 3. Parameters
1. _huggingface_id_ (string, optional)

    The string can be any Hugging Face id from the [Hugging Face models webpage](https://huggingface.co/models)
    
    > Models from Hugging Face are subject to third party license terms available on the Hugging Face model details page. It is your responsibility to comply with the model's license terms.

**NOTE** The _pytorch_model_path_ or _mlflow_model_path_ takes precedence over _huggingface_id_
