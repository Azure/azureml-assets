"""
Tests running the pytorch_benchmark/helper/model/ loader.
"""
import pytest
import torch

from pytorch_benchmark.classification.model import (
    get_model_metadata,
    load_model,
)

# IMPORTANT: see conftest.py for fixtures

# IMPORTANT: we have to restrict the list of models for unit test
# because github actions runners have 7GB RAM only and will OOM
TEST_MODEL_ARCH_LIST = [
    "test",
    "resnet18",
    "resnet34",
]


@pytest.mark.parametrize("model_arch", TEST_MODEL_ARCH_LIST)
def test_model_loader(model_arch):
    "Tests src/components/pytorch_benchmark/helper/model/"
    model_metadata = get_model_metadata(model_arch)

    assert model_metadata is not None
    assert isinstance(model_metadata, dict)
    assert "library" in model_metadata
    assert "input_size" in model_metadata

    # using pretrained=False to avoid downloading each time we unit test
    model = load_model(model_arch, output_dimension=4, pretrained=False)

    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_model_loader_failure():
    "Test asking for a model that deosn't exist"
    with pytest.raises(NotImplementedError):
        get_model_metadata("not_a_model")

    with pytest.raises(NotImplementedError):
        load_model("not_a_model", output_dimension=4, pretrained=False)
