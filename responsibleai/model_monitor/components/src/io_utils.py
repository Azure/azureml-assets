# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to read data."""

import mltable
import pandas as pd


def load_mltable_to_df(mltable_path) -> pd.DataFrame:
    """Load MLTable into a DataFrame.

        :param mltable_path: Path to MLtable
        :type mltable_path string
        :return datafrom loaded from mltable
        :rtype pandas.Dataframe

    """
    return mltable.load(mltable_path).to_pandas_dataframe()
