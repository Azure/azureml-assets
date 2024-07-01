# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Set up azureml-assets package."""

from setuptools import setup, find_packages

setup(
   name="azureml-assets",
<<<<<<< HEAD
   version="1.16.42",
=======
   version="1.16.53",
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
   description="Utilities for publishing assets to Azure Machine Learning system registries.",
   author="Microsoft Corp",
   packages=find_packages(),
   include_package_data=True,
   install_requires=[
      "GitPython>=3.1",
      "ruamel.yaml==0.17.21",
      "pip>=21",
      "marshmallow>=3.19",
      "tenacity>=8.2.2",
      "azure-ai-ml>=1.9.0",
   ],
   python_requires=">=3.8,<4.0",
   license="MIT",
   classifiers=[
      "Development Status :: 5 - Production/Stable",
      "Intended Audience :: Developers",
      "Topic :: Software Development :: Build Tools",
      "License :: OSI Approved :: MIT License",
      "Programming Language :: Python :: 3.8",
      "Programming Language :: Python :: 3.9",
      "Programming Language :: Python :: 3.10",
      "Programming Language :: Python :: 3.11",
   ],
   project_urls={
        # 'Documentation': "https://example.com/documentation/",
        'Source': "https://github.com/Azure/azureml-assets/",
        'Changelog': "https://github.com/Azure/azureml-assets/blob/main/scripts/azureml-assets/CHANGELOG.md",
    },
)
