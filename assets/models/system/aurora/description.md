Aurora is a machine learning model that can predict general environmental variables, such as temperature and wind speed.
It is a _foundation model_, which means that it was first generally trained on a lot of data,
and then can be adapted to specialised environmental forecasting tasks with relatively little data.
We provide four such specialised versions:
one for medium-resolution weather prediction,
one for high-resolution weather prediction,
one for air pollution prediction,
and one for ocean wave prediction.
Currently, this implementation only includes the version for medium-term weather prediction.
Please see the documentation of the Aurora Foundry Python API linked below for
precisely which models are available.

## Resources

* [Documentation of the Aurora Foundry Python API](https://microsoft.github.io/aurora/foundry/intro.html)
* [A full-fledged example that runs the model on Foundry](https://microsoft.github.io/aurora/foundry/demo_hres_t0.html).
* [Implementation of the Aurora model](https://github.com/microsoft/aurora)
* [Documentation of the Aurora implementation](https://microsoft.github.io/aurora/intro.html)
* [Paper with detailed evaluation](https://arxiv.org/abs/2405.13063)

## Quickstart

```python
from aurora import Batch

from aurora.foundry import BlobStorageChannel, FoundryClient, submit


initial_condition = Batch(...)  # Create initial condition for the model.

for pred in submit(
    initial_condition,
    model_name="aurora-0.25-finetuned",
    num_steps=4,  # Every step predicts six hours ahead.
    foundry_client=FoundryClient(
        endpoint="https://endpoint/",
        token="ENDPOINT_TOKEN",
    ),
    # Communication with the endpoint happens via an intermediate blob storage container. You
    # will need to create one and generate an URL with a SAS token that has both read and write
    # rights.
    channel=BlobStorageChannel(
        "https://storageaccount.blob.core.windows.net/container?<READ_WRITE_SAS_TOKEN>"
    ),
):
    pass  # Do something with `pred`, which is a `Batch`.
```
