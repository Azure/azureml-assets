# Model Registration Component
This component can be used in Azure Machine Learning pipelines to register the model in workspace or registry.
Model registration allows you to store and version your models in the Azure cloud, in your workspace. It helps you organize and keep track of your trained models.

# 1. Inputs

1. _model_path_ (URI_FOLDER, required)

    Path to the model directory, it will contain all the files related to model.

2. _model_download_metadata_ (URI_FILE, optional)

    A JSON file which contains information related to model download.

3. _model_metadata_ (URI_FILE, optional)

    YAML/JSON file that contains model metadata confirming to Model V2 contract.
    
4. _model_import_job_path_ (URI_FILE, optional)

    JSON file that contains the job path of model to have lineage. It will help to track the job from which model has been created. 

# 2. Outputs

1. _registration_details_ (URI_FILE)

    Json file into which model registration details will be written. It will also contain the metadata
    of registered model

# 3. Parameters

1. _model_name_ (STRING, optional)

    Model name to use in the registration. If name already exists, the version will be auto incremented. If this parameter will not be passed, script will try to get model_name from _model_download_metadata_ file

2. _model_version_ (STRING, optional)

    Model version in workspace/registry. If the same model name and version exists or if this parameter is will not be passed, the version will be auto incremented.

3. _model_type_ (STRING, optional)

    Type of model that you want to register in workspace. Default value is "mlflow_model".
    Currently we support following model types

    1. mlflow_model
    2. custom_model

4. _model_description_ (STRING, optional)

    Description of the model that will be shown in model card of registered model in AzureML registry or workspace.

5. _registry_name_ (STRING, optional)

    Name of the AzureML asset registry where the model will be registered. Model will be registered in a workspace if this is unspecified.

# 4. Run Settings

This setting helps to choose the compute for running the component code.

> Select *Use other compute target*

- Under this option, you can select either `compute_cluster` or `compute_instance` as the compute type and the corresponding instance / cluster created in your workspace.
- If you have not created the compute, you can create the compute by clicking the `Create Azure ML compute cluster` link that's available while selecting the compute.
- We generally recommend to use Standard_DS3_v2 compute for this node.

