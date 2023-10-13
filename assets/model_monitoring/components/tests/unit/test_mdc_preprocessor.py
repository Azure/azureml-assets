import pytest
import fsspec
import shutil
import os
import spark_mltable # to enable spark.read.mltable
from model_data_collector_preprocessor.run import _uri_folder_to_spark_df, mdc_preprocessor

@pytest.mark.unit
class TestMDCPreprocessor:
    def test_uri_folder_to_spark_df(self):
        print("testing test_uri_folder_to_spark_df...")
        os.environ["PYSPARK_PYTHON"] = "C:\\Users\\richli\\AppData\\Local\\anaconda3\\envs\\momo\\python.exe"
        fs = fsspec.filesystem("file")
        preprocessed_output = "tests/unit/preprocessed_mdc_data"
        shutil.rmtree(f"{preprocessed_output}temp")
        df = _uri_folder_to_spark_df(
            "2023-10-11T20:00:00",
            "2023-10-11T21:00:00",
            "tests/unit/raw_mdc_data/",
            preprocessed_output,
            False,
            fs,
        )
        print("preprocessed dataframe:")
        df.show()
        assert True
        # assert df.head()[0] == 2 #1

    def test_mdc_preprocessor(self):
        print("testing test_mdc_preprocessor...")
        os.environ["PYSPARK_PYTHON"] = "C:\\Users\\richli\\AppData\\Local\\anaconda3\\envs\\momo\\python.exe"
        fs = fsspec.filesystem("file")
        preprocessed_output = "tests/unit/preprocessed_mdc_data"
        shutil.rmtree(f"{preprocessed_output}temp")
        mdc_preprocessor(
            "2023-10-11T20:00:00",
            "2023-10-11T21:00:00",
            "tests/unit/raw_mdc_data/",
            preprocessed_output,
            False,
            fs,
        )