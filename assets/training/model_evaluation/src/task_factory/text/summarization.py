# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Text Summarization."""

from task_factory.base import PredictWrapper
from logging_utilities import get_logger


logger = get_logger(name=__name__)


class Summarizer(PredictWrapper):
    """Summarizer.

    Args:
        PredictWrapper (_type_): _description_
    """

    def predict(self, X_test, **kwargs):
        """Predict.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._ensure_base_model_input_schema(X_test)
        try:
            y_pred = self.model.predict(X_test, **kwargs)
        except TypeError:
            y_pred = self.model.predict(X_test)
        except RuntimeError as re:
            device = kwargs.get("device", -1)
            model_device = self._get_model_device()
            logger.info("Failed on GPU with error: "+repr(re))
            passed_on_device = False
            if device is None and model_device is not None:
                logger.info("Trying with model device.")
                kwargs["device"] = model_device.index
                if kwargs["device"] is not None:
                    try:
                        y_pred = self.model.predict(X_test, **kwargs)
                        passed_on_device = True
                    except:
                        pass
            if passed_on_device:
                return y_pred
            if device != -1:
                logger.warning("Predict failed on GPU. Falling back to CPU")
                kwargs["device"] = -1
                self._ensure_model_on_cpu()
                y_pred = self.model.predict(X_test, **kwargs)
            else:
                raise re

        return y_pred
