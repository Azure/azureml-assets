# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Set up azureml-assets package."""

from setuptools import setup, find_packages

setup(
   name="azureml-assets",
   version="0.0.1",
   description="Scripts that support Azure Machine Learning assets.",
   author="Microsoft Corp",
   packages=find_packages(),
   install_requires=[
      "GitPython>=3.1",
      "pyyaml>=5",
      "pip>=21",
   ],
   python_requires=">=3.8,<4.0",
)
