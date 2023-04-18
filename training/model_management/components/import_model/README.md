# Model Import Pipeline Component
This is pipeline component, which can be used to create an azure machine learning pipelines to import a publicly available model from sources like HuggingFace,Github to user's workspace or registry.

# 1. Inputs

## 1.1 Inputs for Ml-flow Converter Component

1. _license_file_path_ (URI_FILE, optional)

    Path to the license file of the model(which user want to import).

## 1.2 Inputs for Model Registration Component

1. _model_metadata_ (URI_FILE, optional)

    A JSON or a YAML file that contains model metadata confirming to Model V2 contract.

# 2. Outputs

1. _mlflow_model_folder_ (URI_FOLDER)

    Output path of the converted MLFlow model, that user want to import in workspace or registry using this pipeline component.

2. _model_registration_details_ (URI_FILE)

    Json file into which model registration details will be written. It will also contain the metadata
    of registered model.

# 3. Parameters

## 3.1 Parameters for compute

1. _compute_ (STRING, required)

    Compute to run pipeline job.

## 3.2 Parameters for Model Download Component

1. _model_source_ (STRING, optional)

    Source from which model can be downloaded. Currently users can download model from the sources given below. Default value is "Huggingface".

    1. AzureBlob
    2. GIT
    3. Huggingface

2. _model_id_ (STRING, required)

    A valid model id for the model source selected. For example you can specify `bert-base-uncased` for importing HuggingFace bert base uncased model. Please specify the complete URL if **GIT** or **AzureBlob** is selected in `model_source`.

## 3.3 Parameters for Ml-flow Converter Component

1. _task_name_ (STRING, optional)

    A Hugging face task on which model was trained on. A required parameter for transformers mlflow flavor. Can be provided as input here or it will consume from model_download_metadata JSON file(output of [Model Download Component](https://github.com/Azure/azureml-assets/tree/hrishikesh/ref-docs-modelmgmt/training/model_management/components/download_model)). Tasks that we currently supports are listed below

    1. text-classification
    2. fill-mask
    3. token-classification
    4. question-answering
    5. summarization
    6. text-generation
    7. text-classification
    8. translation
    9. image-classification
    10. text-to-image    

## 3.4 Parameters for Model Registration Component

1. _custom_model_name_ (STRING, optional)

    Model name to use in the registration. If name already exists, the version will be auto incremented.
    Can be provided as input here or it will consume from model_download_metadata JSON file(output of [Model Download Component](https://github.com/Azure/azureml-assets/tree/hrishikesh/ref-docs-modelmgmt/training/model_management/components/download_model)).

2. _model_version_ (STRING, optional)

    Model version in workspace/registry. If the same model name and version exists or if this parameter is will not be passed, the version will be auto incremented.

3. _model_description_ (STRING, optional)

    Description of the model that will be shown in model card of registered model in AzureML registry or workspace. Can be provided as input here or it will consume from model_metadata file.
    
4. _registry_name_ (STRING, optional)

    Name of the AzureML asset registry where the model will be registered. Model will be registered in a workspace if this is unspecified.


# 4. Run Settings

This setting helps to choose the compute for running the component code.

> Select *Use other compute target*

- Under this option, you can select either `compute_cluster` or `compute_instance` as the compute type and the corresponding instance / cluster created in your workspace.
- If you have not created the compute, you can create the compute by clicking the `Create Azure ML compute cluster` link that's available while selecting the compute.
- We generally recommend to use Standard_DS3_v2 compute for this node.

