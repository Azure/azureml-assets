"""
Tests running the pytorch_image_classifier/model/ loader
"""
import pytest
import tensorflow as tf

from components.tensorflow_image_segmentation.tf_helper.model import (
    MODEL_ARCH_LIST,
    get_model_metadata,
    load_model,
)

# IMPORTANT: see conftest.py for fixtures


@pytest.mark.parametrize("model_arch", MODEL_ARCH_LIST)
def test_model_loader(model_arch):
    """Tests src/components/pytorch_image_classifier/model/"""
    model_metadata = get_model_metadata(model_arch)

    assert model_metadata is not None
    assert isinstance(model_metadata, dict)
    assert "library" in model_metadata

    model = load_model(model_arch, input_size=160, num_classes=3)

    assert model is not None
    assert isinstance(model, tf.keras.Model)


def test_model_loader_failure():
    """Test asking for a model that deosn't exist"""
    with pytest.raises(NotImplementedError):
        get_model_metadata("not_a_model")

    with pytest.raises(NotImplementedError):
        load_model("not_a_model", input_size=160, num_classes=3)
