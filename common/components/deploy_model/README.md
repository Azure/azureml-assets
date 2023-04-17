# Model Deployment Component
This component can be used in [azure machine learning pipelines](https://learn.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines?view=azureml-api-2) to [deploy](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=azure-cli) the model in [workspace](https://learn.microsoft.com/en-us/azure/machine-learning/concept-workspace?view=azureml-api-2).

The components can be seen here ![as shown in the figure]

# 1. Inputs

1. _registration_details_ (URI_FILE, required)

    Json file that contains all the details of registered model that needs to be deployed.


# 2. Outputs

1. _model_deployment_details_ (URI_FILE)

    Json file to which deployment details will be written.

# 3. Parameters
    
## 3.1 Endpoint parameters
1. _endpoint_name_ (string, optional)

    Name of the (endpoint)[https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints?view=azureml-api-2#managed-online-endpoints-vs-kubernetes-online-endpoints].Endpoint is an HTTPS path that provides an interface for clients to send requests (input data) and receive the inferencing (scoring) output of a trained model.

## 3.2 Deployment parameters
1. _deployment_name_ (string, optional)

    Name of the deployment.User can deploy multiple models at same endpoint with different names and distrbute incoming traffic amongst them.

2. _instance_type_ (string, optional)

    Compute instance type to deploy model. Make sure that instance type is available and have enough quota available.Currently we support following compute types

    1. Standard_DS1_v2
    2. Standard_DS2_v2
    3. Standard_DS3_v2
    4. Standard_DS4_v2
    5. Standard_DS5_v2
    6. Standard_F2s_v2
    7. Standard_F4s_v2
    8. Standard_F8s_v2
    9. Standard_F16s_v2
    10. Standard_F32s_v2
    11. Standard_F48s_v2
    12. Standard_F64s_v2
    13. Standard_F72s_v2
    14. Standard_FX24mds
    15. Standard_FX36mds
    16. Standard_FX48mds
    17. Standard_E2s_v3
    18. Standard_E4s_v3
    19. Standard_E8s_v3
    20. Standard_E16s_v3
    21. Standard_E32s_v3
    22. Standard_E48s_v3
    23. Standard_E64s_v3
    24. Standard_NC4as_T4_v3
    25. Standard_NC6s_v2
    26. Standard_NC6s_v3
    27. Standard_NC8as_T4_v3
    28. Standard_NC12s_v2
    29. Standard_NC12s_v3
    30. Standard_NC16as_T4_v3
    31. Standard_NC24s_v2
    32. Standard_NC24s_v3
    33. Standard_NC64as_T4_v3
    34. Standard_ND40rs_v2
    35. Standard_ND96asr_v4
    36. Standard_ND96amsr_A100_v4

3. _instance_count_ (string, optional)

    Description of the model that will be shown in model card of registered model in AzureML registry or workspace

4. _max_concurrent_requests_per_instance_ (string, optional)

    Name of the AzureML asset registry where the model will be registered. Model will be registered in a workspace if this is unspecified

5. _request_timeout_ms_ (string, optional)

    Description of the model that will be shown in model card of registered model in AzureML registry or workspace

6. _max_queue_wait_ms_ (string, optional)

    Name of the AzureML asset registry where the model will be registered. Model will be registered in a workspace if this is unspecified

## 3.3 Readiness Probe Parameters
1. _failure_threshold_readiness_probe_ (string, optional)

    Description of the model that will be shown in model card of registered model in AzureML registry or workspace

2. _success_threshold_readiness_probe_ (string, optional)

    Name of the AzureML asset registry where the model will be registered. Model will be registered in a workspace if this is unspecified

3. _timeout_readiness_probe_ (string, optional)

    Description of the model that will be shown in model card of registered model in AzureML registry or workspace

4. _period_readiness_probe_ (string, optional)

    Name of the AzureML asset registry where the model will be registered. Model will be registered in a workspace if this is unspecified

5. _initial_delay_readiness_probe_ (string, optional)

    Description of the model that will be shown in model card of registered model in AzureML registry or workspace

## 3.4 Liveness probe parameters
1. _failure_threshold_liveness_probe_ (string, optional)

    Name of the AzureML asset registry where the model will be registered. Model will be registered in a workspace if this is unspecified

2. _timeout_liveness_probe_ (string, optional)

    Name of the AzureML asset registry where the model will be registered. Model will be registered in a workspace if this is unspecified

3. _period_liveness_probe_ (string, optional)

    Description of the model that will be shown in model card of registered model in AzureML registry or workspace

4. _initial_delay_liveness_probe_ (string, optional)

    Name of the AzureML asset registry where the model will be registered. Model will be registered in a workspace if this is unspecified

## 3.5 parameters
1. _egress_public_network_access_ (string, optional)

    Name of the AzureML asset registry where the model will be registered. Model will be registered in a workspace if this is unspecified

# 4. Run Settings

This setting helps to choose the compute for running the component code.

> Select *Use other compute target*

- Under this option, you can select either `compute_cluster` or `compute_instance` as the compute type and the corresponding instance / cluster created in your workspace.
- If you have not created the compute, you can create the compute by clicking the `Create Azure ML compute cluster` link that's available while selecting the compute. See the figure below
![other compute target](https://aka.ms/azureml-ft-docs-create-compute-target)
- We generally recommend to use Standard_DS3_v2 compute for this node.

