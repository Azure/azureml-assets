# Ml-flow Converter Component
This component can be used in [azure machine learning pipelines](https://learn.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines?view=azureml-api-2) to convert the model type to [mlflow](https://www.mlflow.org/docs/latest/models.html).

The components can be seen here ![as shown in the figure]

# 1. Inputs

1. _model_path_ (URI_FILE, required)

    Path to the model for which we want to convert the type to mlflow.

2. _model_download_metadata_ (URI_FILE, optional)

    JSON file containing model download details. File would contain details that could be useful for model registration in forms of model tags and properties

3. _license_file_path_ (URI_FILE, optional)

    Path to the license file of the model. 

# 2. Parameters

1. _model_id_ (string, optional)

    A valid model id for the model source selected. For example you can specify `bert-base-uncased` for importing HuggingFace bert base uncased model. Please specify the complete URL if **GIT** or **AzureBlob** is selected in `model_source`. Can be provided as input here or in model_download_metadata JSON file.


2. _task_name_ (string, optional)

    A Hugging face task on which model was trained on. A required parameter for transformers mlflow flavor. Can be provided as input here or in model_download_metadata JSON file. Tasks that we currently supports are listed below

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

# 3. Outputs

1. _mlflow_model_folder_ (URI_FOLDER)

    Output path for the converted MLFlow model.
    
2. _model_import_job_path_ (URI_FILE)

    JSON file containing model job path for model lineage. This will help user to track the job (while seeing registered model in workspace) that converts the model's type to mlflow.


# 4. Run Settings

This setting helps to choose the compute for running the component code.

> Select *Use other compute target*

- Under this option, you can select either `compute_cluster` or `compute_instance` as the compute type and the corresponding instance / cluster created in your workspace.
- If you have not created the compute, you can create the compute by clicking the `Create Azure ML compute cluster` link that's available while selecting the compute. See the figure below
![other compute target](https://aka.ms/azureml-ft-docs-create-compute-target)
- We generally recommend to use Standard_DS3_v2 compute for this node.

