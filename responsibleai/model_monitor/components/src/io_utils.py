# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to read data."""

import mltable
import os
import pandas as pd
import uuid


def load_mltable_to_df(mltable_path) -> pd.DataFrame:
    """Load MLTable into a DataFrame.

        :param mltable_path: Path to MLtable
        :type mltable_path: string
        :return: datafrom loaded from mltable
        :rtype: pandas.Dataframe

    """
    return mltable.load(mltable_path).to_pandas_dataframe()


def save_df_as_mltable(df: pd.DataFrame, folder_path: str):
    """Write a dataframe to MLTable file.

    Args:
        df: The DataFrame to save.
        folder_path: The folder path in Azure blob store to save the MLTable to.

    """
    filename = f'./{uuid.uuid4()}.parquet'

    df.to_parquet(os.path.join(folder_path, filename), index=False)
    mltable.from_parquet_files(paths=[{'file': filename}]).save(path=folder_path)
