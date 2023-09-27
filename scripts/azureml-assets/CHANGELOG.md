## 1.17.0 (Unreleased)
### 🚀 New Features

### 🐛 Bugs Fixed

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
