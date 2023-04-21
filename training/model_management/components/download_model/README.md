# Model Download Component
This component can be used in Azure Machine Learning pipelines to download the publicy available model.

# 1. Parameters

1. _model_source_ (STRING, required)

    Source from which model can be downloaded. Currently users can download model from the sources given below. Default value is "Huggingface".

    1. AzureBlob
    2. GIT
    3. Huggingface

2. _model_id_ (STRING, required)

    A valid model id for the model source selected. For example you can specify `bert-base-uncased` for importing HuggingFace bert base uncased model. Please specify the complete URL if **GIT** or **AzureBlob** is selected in `model_source`. 

# 2. Outputs

1. _model_output_ (URI_FOLDER)

    Path to the dowloaded model.
    
2. _model_download_metadata_ (URI_FILE)

    File name to which model download details will be written. File would contain details that could be useful for model registration in forms of model tags and properties


# 3. Run Settings

This setting helps to choose the compute for running the component code.

> Select *Use other compute target*

- Under this option, you can select either `compute_cluster` or `compute_instance` as the compute type and the corresponding instance / cluster created in your workspace.
- If you have not created the compute, you can create the compute by clicking the `Create Azure ML compute cluster` link that's available while selecting the compute.
- We generally recommend to use Standard_DS3_v2 compute for this node.

