## Resources

* [Documentation of the Aurora Foundry Python API](https://microsoft.github.io/aurora/foundry/intro.html)
* [A full-fledged example that runs the model on Foundry](https://microsoft.github.io/aurora/foundry/demo_hres_t0.html).
* [Implementation of the Aurora model](https://github.com/microsoft/aurora)
* [Documentation of the Aurora implementation](https://microsoft.github.io/aurora/intro.html)
* [Paper with detailed evaluation](https://arxiv.org/abs/2405.13063)

## Quickstart

First install the model:

```bash
pip install microsoft-aurora
```

Then you can make predictions with a Azure Foundry AI endpoint as follows:

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

## License

This model and the associated model weights are released under the [MIT licence](https://spdx.org/licenses/MIT).

## Security

See [SECURITY](https://github.com/microsoft/aurora/blob/main/SECURITY.md).

## Responsible AI Transparency Documentation

An AI system includes not only the technology, but also the people who will use it, the people who will be affected by it, and the environment in which it is deployed.
Creating a system that is fit for its intended purpose requires an understanding of how the technology works, its capabilities and limitations, and how to achieve the best performance.
Microsoft has a broad effort to put our AI principles into practice.

To find out more, see [Responsible AI principles from Microsoft](https://www.microsoft.com/en-us/ai/responsible-ai).

### Limitations

Although Aurora was trained to accurately predict future weather, air pollution, and ocean waves,
Aurora is based on neural networks, which means that there are no strict guarantees that predictions will always be accurate.
Altering the inputs, providing a sample that was not in the training set,
or even providing a sample that was in the training set but is simply unlucky may result in arbitrarily poor predictions.
In addition, even though Aurora was trained on a wide variety of data sets,
it is possible that Aurora inherits biases present in any one of those data sets.
A forecasting system like Aurora is only one piece of the puzzle in a weather prediction pipeline,
and its outputs are not meant to be directly used by people or businesses to plan their operations.
A series of additional verification tests are needed before it can become operationally useful.

### Data

The models included in the code have been trained on a variety of publicly available data.
A description of all data, including download links, can be found in [Supplementary C of the paper](https://arxiv.org/pdf/2405.13063).
The checkpoints include data from ERA5, CMCC, IFS-HR, HRES T0, GFS T0, and GFS forecasts.

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
