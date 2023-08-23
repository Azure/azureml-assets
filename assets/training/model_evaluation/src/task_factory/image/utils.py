# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Image utils."""

import pandas as pd
import numpy as np
from typing import Union


def convert_predictions(preds: Union[pd.DataFrame, pd.Series, list, np.ndarray]) -> np.ndarray:
    """Convert predictions to numpy array.

    Args:
        predictions(pd.DataFrame, pd.Series, list, np.ndarray): predictions

    Returns:
        np.ndarray: numpy array of predictions
    """
    if isinstance(preds, pd.DataFrame) and len(preds.columns) == 1:
        return preds[preds.columns[0]].to_numpy()
    if isinstance(preds, pd.DataFrame) or isinstance(preds, pd.Series):
        return preds.to_numpy()
    if isinstance(preds, list):
        return np.array(preds)
    return preds
