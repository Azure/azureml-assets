# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from setuptools import setup, find_packages

setup(
   name="azureml-assets",
   version="0.0.1",
   description="Scripts that support Azure Machine Learning assets.",
   author="Microsoft Corp",
   packages=find_packages(),
   install_requires=[
      "GitPython==3.1.27",
      "pyyaml==5.4",
      "requests==2.27.1",
   ],
   python_requires=">=3.6,< 4.0",
)
