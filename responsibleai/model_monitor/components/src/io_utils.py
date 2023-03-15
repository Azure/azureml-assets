# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to read write data."""

import mltable
import pandas as pd

def load_mltable_to_df(mltable_path: str) -> pd.DataFrame:
    """Load MLTable into a DataFrame.

    Args:
        mltable_path: Path to MLtable

    Returns:
        pd.DataFrame: DataFrame loaded from MLtable
    """
    return mltable.load(mltable_path).to_pandas_dataframe()