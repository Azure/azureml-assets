# Mlflow Model Local Validation Component
This component can be used in [azure machine learning pipelines](https://learn.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines?view=azureml-api-2) to valdate if a MLFLow model can be loaded on a compute and is usable for inferencing

The components can be seen here ![as shown in the figure]

# 1. Inputs

1. _model_path_ (MLFLOW_MODEL, required)

    Path to the MLFlow model that needs to be validated.

2. _test_data_path_ (URI_FILE, optional)

    Test dataset for model inferencing.

# 2. Outputs

1. _mlflow_model_folder_ (URI_FOLDER)

    Validated input model. Here input model is used to block further steps in pipeline job if local validation fails.
 
# 3. Parameters

1. _column_rename_map_ (string, optional)

    Provide mapping of dataset column names that should be renamed before inferencing.
    eg: _col1:ren1; col2:ren2; col3:ren3_.

# 4. Run Settings

This setting helps to choose the compute for running the component code.

> Select *Use other compute target*

- Under this option, you can select either `compute_cluster` or `compute_instance` as the compute type and the corresponding instance / cluster created in your workspace.
- If you have not created the compute, you can create the compute by clicking the `Create Azure ML compute cluster` link that's available while selecting the compute. See the figure below
![other compute target](https://aka.ms/azureml-ft-docs-create-compute-target)
- We generally recommend to use Standard_DS3_v2 compute for this node.

