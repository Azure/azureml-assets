# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for online endpoint model."""


from typing import Optional


class OnlineEndpointModel:
    """Class for online endpoint model."""
    def __init__(self, model: str, model_version: Optional[str], model_type: str):
        if model is not None and model.startswith('azureml:'):
            self._model_name = model.split('/')[-3]
            self._model_path = model
            if model_version is None:
                self._model_version = model.split('/')[-1]
        else:
            self._model_name = model
            self._model_path = None
            self._model_version = model_version
        self._model_type = model_type

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def model_version(self) -> str:
        """Get the model version."""
        return self._model_version

    @property
    def model_type(self) -> str:
        """Get the model type."""
        return self._model_type

    @property
    def model_path(self) -> str:
        """Get the model path."""
        if self._model_path is None:
            self._model_path = 'azureml://registries/azureml-meta/models/{}/versions/{}'.format(
                self._model_name,
                self._model_version
            )
        return self._model_path

    def is_aoai_model(self) -> bool:
        """Check if the model is aoai model."""
        return self._model_type == 'oai'

    def is_oss_model(self) -> bool:
        """Check if the model is llama model."""
        return self._model_type == 'oss'
