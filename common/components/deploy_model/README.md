# Model Deployment Component
This component can be used in Azure Machine Learning pipelines to deploy the model in workspace.

# 1. Inputs

1. _registration_details_ (URI_FILE, required)

    Json file that contains all the details of registered model that needs to be deployed.

# 2. Outputs

1. _model_deployment_details_ (URI_FILE)

    Json file to which deployment details will be written.

# 3. Parameters
    
## 3.1 Endpoint parameters

1. _endpoint_name_ (STRING, required)

    Name of the endpoint. Endpoint is an HTTPS path that provides an interface for clients to send requests (input data) and receive the inferencing (scoring) output of a trained model.

## 3.2 Deployment parameters

1. _deployment_name_ (STRING, required)

    Name of the deployment. User can deploy multiple models at same endpoint with different names and distrbute incoming traffic amongst them.

2. _instance_type_ (STRING, optional)

    Compute instance type to deploy model. Make sure that instance type is available and have enough quota available.Default value is "Standard_F8s_v2".
    Currently we support following compute types

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

3. _instance_count_ (INTEGER, optional)

    Number of instances you want to use for deployment. Make sure instance type have enough quota available.
    Default value is "1".

4. _egress_public_network_access_ (STRING, optional)

    This flag secures the deployment by restricting communication between the deployment and the Azure resources used by it. Set to disabled to ensure that the download of the model, code, and images needed by your deployment are secured with a private endpoint. This flag is applicable only for managed online endpoints. Default value is "enabled"
    Possible values
    1. enabled
    2. disabled

## 3.3 Request Settings Parameters

1. _max_concurrent_requests_per_instance_ (INTEGER, optional)

    Maximum concurrent requests to be handled per instance. Default value is "1".

2. _request_timeout_ms_ (INTEGER, optional)

    Request timeout in milliseconds. Max limit is "90000". Default value is "5000".

3. _max_queue_wait_ms_ (INTEGER, optional)

    The maximum amount of time in milliseconds a request will stay in the queue. Default value is "500".

## 3.4 Readiness Probe Parameters

1. _failure_threshold_readiness_probe_ (INTEGER, optional)

    Number of times system will try after failing the readiness probe. Defaut value is "10".

2. _success_threshold_readiness_probe_ (INTEGER, optional)

    The minimum consecutive successes for the readiness probe to be considered successful after failing.Default value is "1".
 

3. _timeout_readiness_probe_ (INTEGER, optional)

    The number of seconds after which the readiness probe times out. Default value is "10".

4. _period_readiness_probe_ (INTEGER, optional)

    How often (in seconds) to perform the readiness probe. Default value is "10".

5. _initial_delay_readiness_probe_ (INTEGER, optional)

    The number of seconds after the container has started before the readiness probe is initiated. Default value is "10".

## 3.5 Liveness probe parameters

1. _failure_threshold_liveness_probe_ (INTEGER, optional)

    No of times system will try after failing the liveness probe. Default value is "30"

2. _timeout_liveness_probe_ (INTEGER, optional)

    The number of seconds after which the liveness probe times out. Default value is "10".

3. _period_liveness_probe_ (INTEGER, optional)

    How often (in seconds) to perform the liveness probe. Default value is "10".

4. _initial_delay_liveness_probe_ (STRING, optional)

    The number of seconds after the container has started before the liveness probe is initiated. Default value is "10".


# 4. Run Settings

This setting helps to choose the compute for running the component code.

> Select *Use other compute target*

- Under this option, you can select either `compute_cluster` or `compute_instance` as the compute type and the corresponding instance / cluster created in your workspace.
- If you have not created the compute, you can create the compute by clicking the `Create Azure ML compute cluster` link that's available while selecting the compute.
- We generally recommend to use Standard_DS3_v2 compute for this node.

