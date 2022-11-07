
### What are the assets being proposed?
To provide provision to the customers to be able to submit LLM training jobs on AML platform using registered components.

**Components**

Currently we have below components on high level viz:
1. Pre-process component
2. Finetune component
3. Inference component
4. Deploy component

Following are the tasks currently supported:
* Single Label Classification
* Multi Label Classification
* Regression
* Named Entity Recognition (NER)

Since there are differences in properties used by above tasks, we are going to register the high level components for each task.

eg: preprocess_classification_multi_label will be component doing pre-processing for multi label classification task

**Environment**

Command component specific environment will be shared, which will contain following assets: 

1. Dockerfile
2. env.yaml


### Why should this asset be built-in?
The goal is to offer fine tuning of large scale models to 1P and 3P users, leveraging the distributed training capabilities in AzureML.

There are primarily 3 model families we focus on:
* Open AI
* Project Alexander
* HuggingFace(+other open source models)

The models have different access restrictions depending on the model family and depending on whether they are exposed to 1P or 3P users


### Support model (what teams will be on the hook for bug fixes and security patches)?
- PM (Swati Gharse: swatig@)
- Finetuning (Naveen Gaur: nagaur@)
- SDK (Sasidhar Kasturi: sasik@)

### A high-level description of the implementation for each asset.

**Component**

1. Pre-process component: The goal of preprocessing component is to validate and tokenize the user data and save the tokenized data and other relevant metadata in the blobstore. 
   1. Inputs to this component are unprocessed train and validation data.
   2. Output of this component will contain tokenized output of the train and validation data. Also, the pretrained model downloaded from the blobstore is copied to the `output_dir` to be made available for finetuning.

2. Finetune component: the goal of finetuning component is to perform finetuning of pretrained models on custom or pre-available datasets. The component supports LoRa, Deepspeed and ONNXRuntime configurations for performance enhancement. 
   1. Inputs to the component are deepspeed config(optional, intended to be passed if `apply_deepspeed` is `true`) and dataset_path (output of preprocessed component)
   2. Outputs of the component are an URI FOLDER cotaining the finetuned model output with checkpoints, model configs, tokenizers

3. Inference component: The goal of this component is to run inference on finetuned model.
   1. Inputs to the component are deepspeed config(optional, intended to be passed if `apply_deepspeed` is `true`),  test file path and model path (path to the output folder containing checkpoints, model configs, tokenizers)
   2. Outputs of the component is path to output directory which contains the generated predictions.txt file containing predictions for the provided test set to the component and other metadata

4. Deploy component: The goal of this component is to create an Managed Online Endpoint of the finetuned model.

**Environment**

1. Dockerfile: Docker config for env. We are making use of PTCA(PyTorch container for Azure) as a base image, on top of which are installing custom packages.
2. env.yaml: environment properties such as name, version will be provided here. Env will be created using Dockerfile mentioned above.
