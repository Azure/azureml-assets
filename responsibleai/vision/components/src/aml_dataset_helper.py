# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import logging
import os
import tempfile
import time
from typing import cast

import azureml.dataprep as dprep

import pandas as pd

from azureml.core import Dataset as AmlDataset
from azureml.data.abstract_dataset import AbstractDataset
from azureml.dataprep import ExecutionError
from azureml.dataprep.api.engineapi.typedefinitions import FieldType
from azureml.dataprep.api.functions import get_portable_path
from azureml.exceptions import UserErrorException

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class AmlDatasetHelper:
    """Helper for AzureML dataset."""

    # AML Dataset Helper is taken from azureml-automl-dnn-vision package.
    # We cannot import azureml-automl-dnn-vision directly since it depends
    # upon python 3.7 and the enviroment has python 3.8 in it.

    # Todo (rupaljain/nvijayrania): Once PTCA support is enabled in runtime,
    # use run time instead of this class, and remove this class.

    LABEL_COLUMN_PROPERTY = '_Label_Column:Label_'
    DEFAULT_LABEL_CONFIDENCE_COLUMN_NAME = 'label_confidence'
    COLUMN_PROPERTY = 'Column'
    IMAGE_COLUMN_PROPERTY = '_Image_Column:Image_'
    DEFAULT_IMAGE_COLUMN_NAME = 'image_url'
    PORTABLE_PATH_COLUMN_NAME = 'PortablePath'
    LOCAL_PATH_COLUMN_NAME = 'image_local_path'
    DATASTORE_PREFIX = 'AmlDatastore://'
    OUTPUT_DATASET_PREFIX = "output_"
    STREAM_INFO_HANDLER_PROPERTY = 'handler'
    DATASTORE_HANDLER_NAME = 'AmlDatastore'

    def __init__(
        self, dataset: AbstractDataset,
        ignore_data_errors: bool = False,
        image_column_name: str = DEFAULT_IMAGE_COLUMN_NAME,
        download_files: bool = True
    ):
        """Constructor - This reads the dataset and downloads the images
        that the dataset contains.

        :param dataset: dataset
        :type dataset: AbstractDataset
        :param ignore_data_errors: Setting this ignores the files in the
        dataset that fail to download.
        :type ignore_data_errors: bool
        :param image_column_name: The column name for the image file.
        :type image_column_name: str
        :param download_files: Flag to download files or not.
        :type download_files: bool
        """

        self._data_dir = AmlDatasetHelper.get_data_dir()

        self.image_col_name = AmlDatasetHelper.get_image_column_name(
                dataset, image_column_name
            )

        if download_files:
            AmlDatasetHelper.download_image_files(
                dataset, self.image_col_name
            )

        dflow = dataset._dataflow.add_column(
            get_portable_path(dprep.col(self.image_col_name)),
            AmlDatasetHelper.PORTABLE_PATH_COLUMN_NAME,
            self.image_col_name
        )

        self.images_df = dflow.to_pandas_dataframe(extended_types=True)

        if download_files and ignore_data_errors:

            missing_file_indices = []
            full_paths = []

            for index in self.images_df.index:
                full_path = self.get_image_full_path(index)
                if not os.path.exists(full_path):
                    missing_file_indices.append(index)
                    msg = "File not found. This file will be ignored"
                    _logger.warning(msg + full_path)
                full_paths.append(full_path)

            self.images_df[self.image_col_name] = full_paths
            self.images_df.drop(missing_file_indices, inplace=True)
            self.images_df.reset_index(inplace=True, drop=True)
            self.images_df = self.images_df.drop(
                [AmlDatasetHelper.PORTABLE_PATH_COLUMN_NAME], axis=1
            )

    def get_image_full_path(self, index: int) -> str:
        """Return the full local path for an image.

        :param index: index
        :type index: int
        :return: Full path for the local image file
        :rtype: str
        """
        return AmlDatasetHelper.get_full_path(
            index, self.images_df, self._data_dir
        )

    @staticmethod
    def get_full_path(
        index: int,
        images_df: pd.DataFrame,
        data_dir: str
    ) -> str:
        """Return the full local path for an image.

        :param index: index
        :type index: int
        :param images_df: DataFrame containing images.
        :type images_df: Pandas DataFrame
        :param data_dir: data folder
        :type data_dir: str
        :return: Full path for the local image file
        :rtype: str
        """
        image_rel_path = images_df[
            AmlDatasetHelper.PORTABLE_PATH_COLUMN_NAME][index]

        # the image_rel_path can sometimes be an exception from dataprep
        if type(image_rel_path) is not str:
            _logger.warning(
                f"The relative path of the image is of type "
                f"{type(image_rel_path)}, expecting a string. "
                f"Will ignore the image."
            )
            image_rel_path = "_invalid_"

        return cast(str, data_dir + '/' + image_rel_path)

    @staticmethod
    def get_data_dir() -> str:
        """Get the data directory to download the image files to.

        :return: Data directory path
        :type: str
        """
        return tempfile.gettempdir()

    @staticmethod
    def _get_column_name(
        ds: AmlDataset,
        parent_column_property: str,
        default_value: str
    ) -> str:
        if parent_column_property in ds._properties:
            image_property = ds._properties[parent_column_property]

            if AmlDatasetHelper.COLUMN_PROPERTY in image_property:
                return cast(
                        str,
                        image_property[AmlDatasetHelper.COLUMN_PROPERTY]
                    )

            lower_column_property = AmlDatasetHelper.COLUMN_PROPERTY.lower()
            if lower_column_property in image_property:
                return cast(str, image_property[lower_column_property])

        return default_value

    @staticmethod
    def get_image_column_name(
        ds: AmlDataset,
        default_image_column_name: str
    ) -> str:
        """Get the image column name by inspecting AmlDataset properties.
        Return default_image_column_name if not found in properties.

        :param ds: Aml Dataset object
        :type ds: TabularDataset (Labeled) or FileDataset
        :param default_image_column_name: default value to return
        :type default_image_column_name: str
        :return: Image column name
        :rtype: str
        """
        return AmlDatasetHelper._get_column_name(
                ds,
                AmlDatasetHelper.IMAGE_COLUMN_PROPERTY,
                default_image_column_name
            )

    @staticmethod
    def is_labeled_dataset(ds: AmlDataset) -> bool:
        """Check if the dataset is a labeled dataset.
        In the current approach, we rely on the presence of
        certain properties to check for labeled dataset.

        :param ds: Aml Dataset object
        :type ds: TabularDataset or TabularDataset (Labeled)
        :return: Labeled dataset or not
        :rtype: bool
        """
        return AmlDatasetHelper.IMAGE_COLUMN_PROPERTY in ds._properties

    @staticmethod
    def download_image_files(ds: AmlDataset, image_column_name: str) -> None:
        """Helper method to download dataset files.

        :param ds: Aml Dataset object
        :type ds: TabularDataset (Labeled) or FileDataset
        :param image_column_name: The column name for the image file.
        :type image_column_name: str
        """
        AmlDatasetHelper._validate_image_column(ds, image_column_name)
        _logger.info("Start downloading image files")
        start_time = time.perf_counter()
        data_dir = AmlDatasetHelper.get_data_dir()
        try:
            if AmlDatasetHelper.is_labeled_dataset(ds):
                ds._dataflow.write_streams(
                    image_column_name,
                    dprep.LocalFileOutput(data_dir)).run_local()
            else:  # TabularDataset
                ds.download(image_column_name, data_dir, overwrite=True)
        except (ExecutionError, UserErrorException) as e:
            raise UserErrorException(
                    f"Could not download dataset files. "
                    f"Please check the logs for more details."
                    f"Error Code: {e}"
                )

        _logger.info(
            "Downloading image files took"
            "{:.2f} seconds".format(time.perf_counter() - start_time)
        )

    @staticmethod
    def _validate_image_column(ds: AmlDataset, image_column_name: str) -> None:
        """Helper method to validate if image column is present in dataset,
         and it's type is STREAM.

        :param ds: Aml Dataset object
        :type ds: TabularDataset (Labeled) or FileDataset
        :param image_column_name: The column name for the image file.
        :type image_column_name: str
        """
        dtypes = ds._dataflow.dtypes
        if image_column_name not in dtypes:
            raise UserErrorException(
                "Image URL column"
                f"'{image_column_name}'"
                " is not present in the dataset."
            )

        image_column_dtype = dtypes.get(image_column_name)
        if image_column_dtype != FieldType.STREAM:
            msg = "The data type of image URL column"
            f"'{image_column_name}' is {image_column_dtype.name}, "
            f"but it should be {FieldType.STREAM.name}."
            raise UserErrorException(msg)
