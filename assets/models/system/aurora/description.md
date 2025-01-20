# Aurora: A Foundation Model for the Earth System

Aurora is a machine learning model that can predict atmospheric variables, such as temperature.
It is a _foundation model_, which means that it was first generally trained on a lot of data,
and then can be adapted to specialised atmospheric forecasting tasks with relatively little data.
We provide four such specialised versions:
one for medium-resolution weather prediction,
one for high-resolution weather prediction,
one for air pollution prediction,
and one for ocean wave prediction.

## Resources

* [Documentation of Aurora  detailed instruction and examples](https://microsoft.github.io/aurora)
* [Documentation of Foundry Python API detailed instruction and examples](https://microsoft.github.io/aurora/foundry/intro.html)
* [A full-fledged Foundry example](https://microsoft.github.io/aurora/foundry/demo_hres_t0.html).
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
