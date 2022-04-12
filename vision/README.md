# Vision Components

This subfolders hosts the code for AzureML assets related to vision machine learning scenarios.

## Structure of this subfolder

| Directory         | Description                                                                          |
|:------------------|:-------------------------------------------------------------------------------------|
| `src/components/` | Python code for the components.                                                      |
| `src/jobs/`       | AzureML CLI jobs for various vision-related tasks.                                   |
| `src/pipelines/`  | AzureML Python SDK code of Azure ML pipelines using components in various scenarios. |
| `tests/`          | Unit tests for the components                                                        |

## For local testing

1. Install requirements

    ```bash
    # to run the component and pipeline scripts locally
    python -m pip install -r ./requirements.txt
    ```

2. Run unit tests

    ```bash
    pytest ./tests
    ```

## Running a pipeline using Azure ML CLI v2

### Requirements

1. Set your `az` CLI with your subscription, workspace and resource group:

    ```bash
    # set the name of your subscription in az cli
    az account set --name "..."

    # set references to connect to Azure ML
    az config set defaults.workspace="..." defaults.resource_group="..."
    ```

2. Create the required environments

    ```bash
    az ml environment create --file ./assets/environments/nvidia/env.yml
    ```

### Run a test pipeline

You can use the az ml cli to run a test pipeline:

```bash
az ml job create --f src/pipelines/canary/classification_random.yml
```

Running this job does not require any particular dataset. The corresponding pipeline will generate 4 image folders with random noise images. Then will train the classifier on this dataset for 5 epochs.


## Running a benchmark pipeline

### Create train/valid datasets using jobs

For running the benchmark pipeline, you'll need to create training and validation datasets first. The following instructions will let you create 2 benchmark datasets:

| Dataset                                                            | Description                                                                         |
|:-------------------------------------------------------------------|:------------------------------------------------------------------------------------|
| [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) | 20580 images in 120 classes (one folder per class).                                 |
| [Places2](http://places2.csail.mit.edu/download.html)              | 1.8 million training images in 365 classes, 18250 validation images (50 per class). |

To create those datasets, you'll need to run jobs that will unpack the dataset archive. Those jobs use the Azure ML CLI v2.

1. Run each job in your workspace. To unpack the Places2 dataset, you will need a SKU with at least 100GB of disk.

    ```bash
    # to use a specific cluster, override with --set
    az ml job create --file src/jobs/create_stanford_dogs_dataset.yml --web

    # to use a specific cluster, override with --set
    az ml job create --file src/jobs/create_places2_dataset.yml --web --set compute="cpu-cluster-d12"
    ```

    **Important** : unpacking Stanford Dogs dataset should take a couple minutes (3mins in our tests), while Places2 might take up to 30-45 mins depending on SKU.

2. Once the jobs complete, go into the Azure ML portal and manually register the outputs with a corresponding name:
    - `stanford_dogs` is only one output we'll use for both training and validation
    - `places2_train` and `places2_valid` as each of the outputs of the job.

    **Work in progress**: automatic registration of dataset within jobs is coming, once it's available you won't need to manually register the datasets yourself.

### Run a benchmark pipeline

To run a training on a given pair of training/validation dataset, use the az ml cli again:

```bash
az ml job create --file src/pipelines/benchmark/train.yml
```

The cli allows you to override all parameters from the command line. Check the content of the yaml to align your override syntax with the tree of yaml fields in the job. For instance, to increase the number of nodes, use:

```bash
az ml job create --file src/pipelines/benchmark/train.yml --set jobs.train.resources.instance_count=2
```

## Run script locally or on a VM

If you want to run this component locally or on a GPU VM to test performance, please follow instructions below.

### Download the Places2 dataset locally

```bash
# create data/ folder (gitignored)
mkdir data

# download the 24.8G archive
curl -o ./data/archive.tar http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar

# alternatively, download the Stanford Dogs (smaller)
# curl -o ./data/archive.tar http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar

# unpack in the data/ folder
# with so many files, do not use verbose, but add checkpoint to show progress
tar xfm ./data/archive.tar --no-same-owner --checkpoint=1000 -C ./data/
```

For the Places2 dataset, it will create 2 subfolders:
- `data/places365_standard/train/` for the training dataset
- `data/places365_standard/val/` for the validation dataset

Alternatively, for the Stanford Dogs dataset, it will create:
- `data/Images/` for the entire dataset (no train/val split)

### Run pytorch distributed from command line

Use the [PyTorch distributed launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility) to run the script on your VM with multiple GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE \
    src/components/pytorch_image_classifier/train.py \
    --train_images ./data/places365_standard/train/ \
    --valid_images ./data/places365_standard/val/ \
    --model_arch resnet18 \
    --num_epochs 5 \
    --batch_size 64 \
    --num_workers 8 \
    --prefetch_factor 2 \
    --distributed_backend nccl \
    --enable_profiling False
```

The script will use mlflow to track its metrics, and log the model. Use [MLflow tracking command line interface](https://mlflow.org/docs/latest/tracking.html) to get the metrics. Or watch for the logs, they will display those metrics inline as well.
