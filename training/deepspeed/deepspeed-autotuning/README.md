### Deepspeed autotuning with Azure Machine Learning
## Overview
The deepspeed autotuner will generate an optimal configuration file (``ds_config.json``) that can be used to achieve good speed in a deepspeed training job.
## How to Run
1. Create a compute that can run the job. Computes with Tesla V100 GPUs provide good compute power. In the ``deepspeed-autotune-aml.yaml`` file, replace ``<name-of-your-compute-here>`` with the name of your compute.
2. Submit the autotuning job with the following command:<br />
```az ml job create --file deepspeed-autotune-aml.yaml```