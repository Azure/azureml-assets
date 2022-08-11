
### What are the assets being proposed?
To provide provision to the customers to be able to submit LLM training jobs on AML platform using registered components.

**Components**

Currently we have below components on high level viz:
1. Pre-process component
2. Finetune component
3. Inference component
4. Deploy component

And, we are supporting tasks, namely:
1. Single label classification
2. Multi label classification
3. Named Entity Recognition (NER)
4. Regression

Since there are differences in properties used by above tasks, we are going to register the high level components for each task.

eg: preprocess_classification_multi_label will be component doing pre-processing for multi label classification task

**Environment**

Command component specific environment will be shared, which will contain following assets: 

1. Dockerfile
2. env.yaml


### Why should this asset be built-in?
The goal is to offer fine tuning of large scale models to 1P and 3P users, leveraging the distributed training capabilities in AzureML.

### Support model (what teams will be on the hook for bug fixes and security patches)?
- PM (Swati Gharse: swatig@)
- Finetuning (Naveen Gaur: nagaur@)
- SDK (Sasidhar Kasturi: sasik@)

### A high-level description of the implementation for each asset.

**Component**

1. Pre-process component: This component will do pre-processing on input data provided.

2. Finetune component: This component will do finetuning on LLM model, type of which will be provided by the customer. Input of this model will be pre-processed data outputted by pre-processing component. Output would be finetuned model.

3. Inference component: This component will perform inferencing on the finetuned model.

4. Deploy component: This component will be used for endpoint creation and deployment of finetuned model.

**Environment**

1. Dockerfile: Docker config for env 
2. env.yaml: environment properties such as name, version will be provided here. Env will be created using Dockerfile mentioned above.

