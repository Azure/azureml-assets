### Deepspeed training with Azure Machine Learning
## Overview
Train a model using deepspeed.
## How to Run
1. Create a compute that can run the job. Computes with Tesla V100 GPUs provide good compute power. In the ``deepspeed-autotune-aml.yaml`` file, replace ``<name-of-your-compute-here>`` with the name of your compute.
2. This example provides a basic ``ds_config.json`` file to configure deepspeed. To have a more optimal configuration, run the deepspeed-autotuning example first to generate a new ds_config file to replace this one.
3. Submit the training job with the following command:<br />
```bash create-job.sh```