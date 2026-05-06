# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Set up azureml-assets package."""

from setuptools import setup, find_packages

setup(
   name="azureml-assets",
   version="1.17.3",
   description="Utilities for publishing assets to Azure Machine Learning system registries.",
   author="Microsoft Corp",
   packages=find_packages(),
   include_package_data=True,
   install_requires=[
      "GitPython>=3.1",
      "ruamel.yaml~=0.18.10",
      "pip>=21",
      "marshmallow>=3.19",
      "tenacity>=8.2.2",
      "azure-ai-ml>=1.9.0",
      "azure-identity>=1.16.0",
      # azure-cli is invoked via subprocess (e.g. publish_utils.py, environment/build.py).
      # Pinned compatible with msftkube[base-envs,templates,packaged-distribution]
      # 2026.4.x+, which requires azure-cli~=2.81.0. Without this declaration the
      # joint pip resolution of azureml-assets + msftkube cannot find a solution
      # (azureml-assets's transitive azure-ai-ml tree fights msftkube's azure-cli
      # tree), causing ResolutionImpossible in pipelines that install both.
      "azure-cli~=2.81.0",
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
      "Programming Language :: Python :: 3.12",
   ],
   project_urls={
        # 'Documentation': "https://example.com/documentation/",
        'Source': "https://github.com/Azure/azureml-assets/",
        'Changelog': "https://github.com/Azure/azureml-assets/blob/main/scripts/azureml-assets/CHANGELOG.md",
    },
)
