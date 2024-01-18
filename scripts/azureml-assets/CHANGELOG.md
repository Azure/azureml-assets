## 1.17.0 (Unreleased)
### 🚀 New Features

### 🐛 Bugs Fixed

## 1.16.28 (2023-01-05)
### 🚀 New Features
- [#2132](https://github.com/Azure/azureml-assets/pull/2132) Improve logging for model validation and link validation results and build running it

## 1.16.28 (2023-01-05)
### 🚀 New Features
- [#2071](https://github.com/Azure/azureml-assets/pull/2071) Allow updating files during azcopy

## 1.16.27 (2023-12-22)
### 🐛 Bugs Fixed
- [#2031](https://github.com/Azure/azureml-assets/pull/2031) Update model validation for shared quota usage

## 1.16.26 (2023-12-21)
### 🐛 Bugs Fixed
- [#2021](https://github.com/Azure/azureml-assets/pull/2021) Require auto version for environments

## 1.16.25 (2023-12-21)
### 🐛 Bugs Fixed
- [#2009](https://github.com/Azure/azureml-assets/pull/2009) Fix call to get tokens

## 1.16.24 (2023-12-20)
### 🐛 Bugs Fixed
- [#2006](https://github.com/Azure/azureml-assets/pull/2006) Make SAS expiration configurable

## 1.16.23 (2023-12-19)
### 🚀 New Features
- [#1909](https://github.com/Azure/azureml-assets/pull/1909) Model spec validation

## 1.16.22 (2023-12-18)
### 🚀 New Features
- [#1980](https://github.com/Azure/azureml-assets/pull/1980) Add switch to skip validating pytest existence

## 1.16.21 (2023-12-18)
### 🚀 New Features
- [#1964](https://github.com/Azure/azureml-assets/pull/1964) Support conda environment YAML for pytest

## 1.16.20 (2023-12-05)
### 🐛 Bugs Fixed
- [#1871](https://github.com/Azure/azureml-assets/pull/1871) Update model validation criteria

## 1.16.19 (2023-12-04)
### 🐛 Bugs Fixed
- [#1875](https://github.com/Azure/azureml-assets/pull/1875) Generate SAS tokens for prompts in addition to models

## 1.16.18 (2023-11-20)
### 🐛 Bugs Fixed
- [#1760](https://github.com/Azure/azureml-assets/pull/1760) Generate SAS tokens for multiple models and output in JSON

## 1.16.17 (2023-11-07)
### 🐛 Bugs Fixed
- [#1703](https://github.com/Azure/azureml-assets/pull/1703) Support for storage account access for model copy with SAS token

## 1.16.16 (2023-11-03)
### 🐛 Bugs Fixed
- [#1643](https://github.com/Azure/azureml-assets/pull/1643) Optimizing SAS token generation when needed to access storage accounts

## 1.16.15 (2023-10-30)
### 🐛 Bugs Fixed
- [#1523](https://github.com/Azure/azureml-assets/pull/1523) Support for storage accounts that are not configured for anonymous access

## 1.16.14
### 🐛 Bugs Fixed
- [#1574](https://github.com/Azure/azureml-assets/pull/1574) Fix fetching validated model assets

## 1.16.13
### 🚀 New Features
- [#1569](https://github.com/Azure/azureml-assets/pull/1569) Add model validations

## 1.16.12 (2023-10-12)
### 🐛 Bugs Fixed
- [#1458](https://github.com/Azure/azureml-assets/pull/1458) Update schema for prompt and benchmark assets

## 1.16.11 (2023-10-09)
### 🐛 Bugs Fixed
- [#1428](https://github.com/Azure/azureml-assets/pull/1428) Azcopy fixes to support additional clouds

## 1.16.10 (2023-10-06)
### 🐛 Bugs Fixed
- [#1411](https://github.com/Azure/azureml-assets/pull/1411) Prevent Config._expand_path from returning directories

## 1.16.9 (2023-10-05)
### 🚀 New Features
- [#1345](https://github.com/Azure/azureml-assets/pull/1345) Add sample prompt assets and prompt publishing

## 1.16.8 (2023-10-05)
### 🐛 Bugs Fixed
- [#1393](https://github.com/Azure/azureml-assets/pull/1393) Fix mlflow model and keep parent mlflow_model_folder


## 1.16.7 (2023-10-03)
### 🐛 Bugs Fixed
- [#1363](https://github.com/Azure/azureml-assets/pull/1363) Fix AzureBlobstoreAssetPath storage account
  URI logic

## 1.16.6 (2023-10-02)
### 🐛 Bugs Fixed
- [#1357](https://github.com/Azure/azureml-assets/pull/1357) Fix validate_assets arg

## 1.16.5 (2023-09-29)
### 🐛 Bugs Fixed
- [#1341](https://github.com/Azure/azureml-assets/pull/1341) Prevent nested directory creation during model upload

## 1.16.4 (2023-09-27)
### 🚀 New Features
- [#1327](https://github.com/Azure/azureml-assets/pull/1327) Support prompt asset type

## 1.16.3 (2023-09-26)
### 🐛 Bugs Fixed
- [#1317](https://github.com/Azure/azureml-assets/pull/1317) Improve sovereign cloud support by removing direct reference to Azure Public Cloud

## 1.16.2 (2023-09-23)
### 🐛 Bugs Fixed
- [#1288](https://github.com/Azure/azureml-assets/pull/1288) Add arg for no-op model updates

## 1.16.1 (2023-09-22)
### 🐛 Bugs Fixed
- [#1288](https://github.com/Azure/azureml-assets/pull/1288) Support to keep latest model version intact

## 1.16.0 (2023-09-12)
### 🚀 New Features
- [#1207](https://github.com/Azure/azureml-assets/pull/1207) Support copy of changed files

## 1.15.2 (2023-09-08)
### 🐛 Bugs Fixed
- [#1188](https://github.com/Azure/azureml-assets/pull/1188) Decrease azcopy verbosity

## 1.15.1 (2023-09-07)
### 🐛 Bugs Fixed
- [#1182](https://github.com/Azure/azureml-assets/pull/1182) Fix authentication when retrieving temporary data references during model upload

## 1.15.0 (2023-09-06)
### 🐛 New Features
- [#1174](https://github.com/Azure/azureml-assets/pull/1174) Support updating model properties

## 1.14.2 (2023-08-31)
### 🐛 Bugs Fixed
- [#1150](https://github.com/Azure/azureml-assets/pull/1147) Fix model registration issue

## 1.14.1 (2023-08-31)
### 🐛 Bugs Fixed
- [#1147](https://github.com/Azure/azureml-assets/pull/1147) Use packaging directly

## 1.14.0 (2023-08-25)
### 🚀 New Features
- [#1098](https://github.com/Azure/azureml-assets/pull/1098) Allow to update Model metadata without increasing the version.

## 1.13.0 (2023-08-11)
### 🚀 New Features
- [#1030](https://github.com/Azure/azureml-assets/pull/1030) Allow "microsoft" to appear in model names

## 1.12.0 (2023-08-07)
### 🚀 New Features
- [#991](https://github.com/Azure/azureml-assets/pull/991) Allow uppercase characters in model asset naming

## 1.11.0 (2023-08-07)
### 🚀 New Features
- [#992](https://github.com/Azure/azureml-assets/pull/992) Extract asset dependencies

## 1.10.0 (2023-08-07)
### 🚀 New Features
- [970](https://github.com/Azure/azureml-assets/pull/970) Add large model publishing support

## 1.9.0 (2023-08-03)
### 🚀 New Features
- [#962](https://github.com/Azure/azureml-assets/pull/962) Prevent curated environment image references in Dockerfiles

## 1.8.0 (2023-07-20)
### 🚀 New Features
- [#913](https://github.com/Azure/azureml-assets/pull/913) Output variable with list of built images

## 1.7.2 (2023-07-20)
### 🐛 Bugs Fixed
- [#912](https://github.com/Azure/azureml-assets/pull/912) Fix indentation in update_assets.py

## 1.7.1 (2023-07-19)
### 🐛 Bugs Fixed
- [#907](https://github.com/Azure/azureml-assets/pull/907) Handle path parameter for data assets

## 1.7.0 (2023-07-18)
### 🚀 New Features
- [#902](https://github.com/Azure/azureml-assets/pull/902) Make output directory optional for update_assets.py

## 1.6.0 (2023-07-17)
### 🚀 New Features
- [#821](https://github.com/Azure/azureml-assets/pull/821) Make release directory optional for update_assets.py

## 1.5.2 (2023-07-13)
### 🐛 Bugs Fixed
- [#882](https://github.com/Azure/azureml-assets/pull/882) Create package list using pip if conda is unavailable

## 1.5.1 (2023-07-11)
### 🚀 New Features
- [#856](https://github.com/Azure/azureml-assets/pull/856) Add ACR task step to output conda export

### 🐛 Bugs Fixed
- [#872](https://github.com/Azure/azureml-assets/pull/872) Be more restrictive when pinning image tags

## 1.5.0 (2023-07-06)
### 🚀 New Features
- [#847](https://github.com/Azure/azureml-assets/pull/847) Add regex arg to skip name validation
- [#801](https://github.com/Azure/azureml-assets/pull/801) Enable vulnerability scanning of environment images

## 1.4.1 (2023-06-28)
### 🐛 Bugs Fixed
- [#802](https://github.com/Azure/azureml-assets/pull/802) Fix version handling when building environment images

## 1.4.0 (2023-06-28)
### 🚀 New Features
- [#800](https://github.com/Azure/azureml-assets/pull/800) Only push images that are configured for publishing

## 1.3.0 (2023-06-27)
### 🚀 New Features
- [#789](https://github.com/Azure/azureml-assets/pull/789) Count assets by type when copying

## 1.2.0 (2023-06-23)
### 🚀 New Features
- [#777](https://github.com/Azure/azureml-assets/pull/777) Make AssetConfig hashable, add DeploymentConfig.should_create()
- [#747](https://github.com/Azure/azureml-assets/pull/747) Allow copy_assets to fail if previous environment version doesn't exist in MCR

### 🐛 Bugs Fixed
- [#745](https://github.com/Azure/azureml-assets/pull/745) Run uniqueness checks even on unchanged assets
- [#770](https://github.com/Azure/azureml-assets/pull/770) Fix resolution of {{latest-image-tag}}

## 1.1.0 (2023-06-08)
### 🚀 New Features
- [#725](https://github.com/Azure/azureml-assets/pull/725) Retry image manifest calls
- [#736](https://github.com/Azure/azureml-assets/pull/736) Add deploy config classes

## 1.0.1 (2023-05-18)
### 🚀 New Features
- [#675](https://github.com/Azure/azureml-assets/pull/675) Read `inference_config` from spec files
- [#678](https://github.com/Azure/azureml-assets/pull/678) Read `os_type` from spec files, create change log

## 1.0.0 (2023-05-16)
### 🚀 New Features
- [#663](https://github.com/Azure/azureml-assets/pull/663) Initial release to PyPI
