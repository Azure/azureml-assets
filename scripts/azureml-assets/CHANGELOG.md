## 1.5.0 (Unreleased)
### 🚀 New Features

### 🐛 Bugs Fixed

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
