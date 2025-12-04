## 1.17.0 (Unreleased)
### 🚀 New Features
### 🐛 Bugs Fixed

## 1.16.99 (2025-11-24)
### 🐛 Bugs Fixed
- [#4636](https://github.com/Azure/azureml-assets/pull/4636) Handle non-JSON content from Azure CLI responses

## 1.16.98 (2025-11-19)
### 🚀 New Features
- [#4629](https://github.com/Azure/azureml-assets/pull/4629) Enforce encoding for file read and write operations

## 1.16.97 (2025-10-22)
### 🚀 New Features
- [#4535](https://github.com/Azure/azureml-assets/pull/4535) Add missing generic asset types to GENERIC_ASSET_TYPES

## 1.16.96 (2025-10-21)
### 🚀 New Features
- [#4533](https://github.com/Azure/azureml-assets/pull/4533) Push images for vulnerability scanning

## 1.16.95 (2025-10-17)
### 🚀 New Features
- [#4528](https://github.com/Azure/azureml-assets/pull/4528) Allow Azure- prefix for models

## 1.16.94 (2025-10-03)
### 🚀 New Features
- [#4483](https://github.com/Azure/azureml-assets/pull/4483) Add APPTEMPLATE to AssetType

## 1.16.93 (2025-10-01)
### 🚀 New Features
- [#4477](https://github.com/Azure/azureml-assets/pull/4477) Onboard AgentManifest asset type

## 1.16.92 (2025-09-26)
### 🚀 New Features
- [#4460](https://github.com/Azure/azureml-assets/pull/4460) Skip validation for unsupported SDK fields

## 1.16.91 (2025-09-19)
### 🚀 New Features
- [#4448](https://github.com/Azure/azureml-assets/pull/4448) Add EVALUATOR to AssetType

## 1.16.90 (2025-09-12)
### 🚀 New Features
- [#4434](https://github.com/Azure/azureml-assets/pull/4434) Support system_metadata by using the SDK for model create and update

## 1.16.89 (2025-08-18)
### 🐛 Bugs Fixed
- [#4391](https://github.com/Azure/azureml-assets/pull/4391) Pass in correct uri to LocalAssetPath from ModelConfig

## 1.16.88 (2025-08-12)
### 🚀 New Features
- [#4383](https://github.com/Azure/azureml-assets/pull/4383) Add ACR task post-push Trivy SBOM scan and SBOM attachment steps

## 1.16.87 (2025-08-08)
### 🚀 New Features
- [#4378](https://github.com/Azure/azureml-assets/pull/4378) Release version 1.16.87

## 1.16.87b1 (2025-08-05)
### 🚀 New Features
- [#4370](https://github.com/Azure/azureml-assets/pull/4370) Bump ruamel.yaml to 0.18.10

## 1.16.86 (2025-07-28)
### 🚀 New Features
- [#4352](https://github.com/Azure/azureml-assets/pull/4352) Detect if MLFlow model is brand new

## 1.16.85 (2025-06-11)
### 🚀 New Features
- [#4256](https://github.com/Azure/azureml-assets/pull/4256) Surface improved error message if job component version is None

## 1.16.84 (2025-05-29)
### 🚀 New Features
- [#4207](https://github.com/Azure/azureml-assets/pull/4207) Detect if MLFlow model was updated

## 1.16.83 (2025-05-22)
### 🐛 Bugs Fixed
- [#4208](https://github.com/Azure/azureml-assets/pull/4208) Fix for python 3.9 - remove union typing

## 1.16.82 (2025-05-22)
### 🚀 New Features
- [#4206](https://github.com/Azure/azureml-assets/pull/4206) Print azcopy log if output_level==default

## 1.16.81 (2025-05-15)
### 🐛 Bugs Fixed
- [#4184](https://github.com/Azure/azureml-assets/pull/4184) Generate SAS token instead of checking container access

## 1.16.80 (2025-05-13)
### 🚀 New Features
- [#4172](https://github.com/Azure/azureml-assets/pull/4172) Make asset validation work with new schema validation in ADO

## 1.16.79 (2025-04-22)
### 🚀 New Features
- [#4117](https://github.com/Azure/azureml-assets/pull/4117) Update spec with modelVariant info after validating

## 1.16.78 (2025-04-21)
### 🚀 New Features
- [#4113](https://github.com/Azure/azureml-assets/pull/4113) Skip validating modelVariant section of asset spec

## 1.16.77 (2025-04-01)
### 🚀 New Features
- [#4029](https://github.com/Azure/azureml-assets/pull/4029) Add AGENTBLUEPRINT to AssetType

## 1.16.76 (2025-03-11)
### 🚀 New Features
- [#3933](https://github.com/Azure/azureml-assets/pull/3933) Allow azcopy output level to be configurable

## 1.16.75 (2025-03-11)
### 🐛 Bugs Fixed
- [#3928](https://github.com/Azure/azureml-assets/pull/3928) Fix validate assets for updated task in quality evaluationresult assets

## 1.16.74 (2025-02-28)
### 🚀 New Features
- [#3887](https://github.com/Azure/azureml-assets/pull/3887) Add support for component and data metadata updates

## 1.16.73 (2025-02-21)
### 🚀 New Features
- [#3853](https://github.com/Azure/azureml-assets/pull/3853) Add H100 GPU to supported_inference_skus.json

## 1.16.72 (2025-02-19)
### 🐛 Bugs Fixed
- [#3846](https://github.com/Azure/azureml-assets/pull/3846) Update account_uri when storage_name is updated for AzureBlobstoreAssetPath

## 1.16.71 (2025-02-12)
### 🚀 New Features
- [#3838](https://github.com/Azure/azureml-assets/pull/3838) Add setter for AzureBlobstoreAssetPath storage_name

## 1.16.70 (2025-02-06)
### 🚀 New Features
- [#3825](https://github.com/Azure/azureml-assets/pull/3825) Pick up latest dependency changes

## 1.16.69 (2025-01-21)
### 🐛 Bugs Fixed
- [#3779](https://github.com/Azure/azureml-assets/pull/3779) Add azure-identity dependency to fix issue with azure-ai-ml dependency

## 1.16.68 (2025-01-02)
### 🚀 New Features
- [#3714](https://github.com/Azure/azureml-assets/pull/3714) Block environment/image releases based on Ubuntu 20.04

## 1.16.67 (2024-11-18)
### 🚀 New Features
- [#3605](https://github.com/Azure/azureml-assets/pull/3605) Improve logging for environment release tag already found in copy_assets

## 1.16.66 (2024-11-11)
### 🚀 New Features
- [#3585](https://github.com/Azure/azureml-assets/pull/3585) Migrate evaluationresult assets to internal repo

## 1.16.65 (2024-11-04)
### 🐛 Bugs Fixed
- [#3544](https://github.com/Azure/azureml-assets/pull/3544) Fix validate assets for new evaluationresult asset tags

## 1.16.64 (2024-10-31)
### 🐛 Bugs Fixed
- [#3549](https://github.com/Azure/azureml-assets/pull/3549) Validate build logs for deprecated dependencies only if logs directory exists

## 1.16.63 (2024-10-29)
### 🚀 New Features
- [#3541](https://github.com/Azure/azureml-assets/pull/3541) Support for new evaluationresult asset tags

## 1.16.62 (2024-10-29)
### 🚀 New Features
- [#3536](https://github.com/Azure/azureml-assets/pull/3536) Validate build logs for deprecated dependencies (Python 3.8)

## 1.16.61 (2024-10-21)
### 🚀 New Features
- [#3514](https://github.com/Azure/azureml-assets/pull/3514) Support publishing Triton models

## 1.16.60 (2024-09-24)
### 🚀 New Features
- [#3431](https://github.com/Azure/azureml-assets/pull/3431) Allow Azure-AI prefix for models

## 1.16.59 (2024-09-24)
### 🚀 New Features
- [#3317](https://github.com/Azure/azureml-assets/pull/3317) Support for storage account access for data asset copy with SAS token

## 1.16.58 (2024-09-13)
### 🐛 Bugs Fixed
- [#3377](https://github.com/Azure/azureml-assets/pull/3377) Validation for vision results

## 1.16.57 (2024-08-12)
### 🐛 Bugs Fixed
- [#3253](https://github.com/Azure/azureml-assets/pull/3253) Also accept Path types for populating tags from files
- [#3254](https://github.com/Azure/azureml-assets/pull/3254) Return string value instead of Path if string doesn't resolve to a path

## 1.16.56 (2024-08-09)
### 🚀 New Features
- [#3232](https://github.com/Azure/azureml-assets/pull/3232) Add additional features to AzureBlobstoreAssetPath
### 🐛 Bugs Fixed
- [#3238](https://github.com/Azure/azureml-assets/pull/3238) Check if value is string before appending to asset filepath

## 1.16.55 (2024-08-08)
### 🚀 New Features
- [#3230](https://github.com/Azure/azureml-assets/pull/3230) Add support for adding tags from filepaths

## 1.16.54 (2024-07-08)
### 🐛 Bugs Fixed
- [#3127](https://github.com/Azure/azureml-assets/pull/3127) Test pipeline

## 1.16.53 (2024-06-28)
### 🐛 Bugs Fixed
- [#3102](https://github.com/Azure/azureml-assets/pull/3102) Don't auth to Azure during import of validate_assets

## 1.16.52 (2024-06-03)
### 🚀 New Features
- [#3000](https://github.com/Azure/azureml-assets/pull/3000) Allow use of azcopy's --overwrite flag

## 1.16.51 (2024-05-30)
### 🐛 Bugs Fixed
- [#2991](https://github.com/Azure/azureml-assets/pull/2991) Generate SAS tokens for data assets

## 1.16.50 (2024-05-22)
### 🐛 Bugs Fixed
- [#2956](https://github.com/Azure/azureml-assets/pull/2956) Update embeddings asset task type

## 1.16.49 (2024-05-17)
### 🐛 Bugs Fixed
- [#2935](https://github.com/Azure/azureml-assets/pull/2935) Update the manifest file to include additional configs

## 1.16.48 (2024-05-17)
### 🐛 Bugs Fixed
- [#2934](https://github.com/Azure/azureml-assets/pull/2934) Update asset validation check for EvaluationResult

## 1.16.47 (2024-05-17)
### 🐛 Bugs Fixed
- [#2933](https://github.com/Azure/azureml-assets/pull/2933) skip hiddenlayerscanned tags till most models are scanned

## 1.16.46 (2024-05-15)
### 🐛 Bugs Fixed
- [#2907](https://github.com/Azure/azureml-assets/pull/2907) add hiddenlayerscanned tags valdn

## 1.16.45 (2024-04-24)
### 🐛 Bugs Fixed
- [#2773](https://github.com/Azure/azureml-assets/pull/2773) Add h100 skus to supported inference skus list

## 1.16.44 (2024-04-19)
### 🐛 Bugs Fixed
- [#2730](https://github.com/Azure/azureml-assets/pull/2730) Fix ACR task timeout keys

## 1.16.43 (2024-04-19)
### 🐛 Bugs Fixed
- [#2726](https://github.com/Azure/azureml-assets/pull/2726) Set overall ACR task timeout

## 1.16.42 (2024-04-03)
### 🐛 Bugs Fixed
- [#2591](https://github.com/Azure/azureml-assets/pull/2637) Exclude yanked version for `latest-pypi-version`

## 1.16.41 (2024-03-27)
### 🐛 Bugs Fixed
- [#2591](https://github.com/Azure/azureml-assets/pull/2591) Revert to not fail on workspace asset URI

## 1.16.40 (2024-03-26)
### 🐛 Bugs Fixed
- [#2585](https://github.com/Azure/azureml-assets/pull/2585) Revert enforcing registry asset URI for components

## 1.16.39 (2024-03-26)
### 🐛 Bugs Fixed
- [#2584](https://github.com/Azure/azureml-assets/pull/2584) Fix empty create configs

## 1.16.38 (2024-03-25)
### 🐛 Bugs Fixed
- [#2572](https://github.com/Azure/azureml-assets/pull/2572) Enforce using registry asset URI for components

## 1.16.37 (2024-03-11)
### 🐛 Bugs Fixed
- [#2476](https://github.com/Azure/azureml-assets/pull/2476) Fix bug for extras with `latest-pypi-version`

## 1.16.36 (2024-03-05)
### 🐛 Bugs Fixed
- [#2282](https://github.com/Azure/azureml-assets/pull/2282) Add support for optional dependencies with `latest-pypi-version`

## 1.16.35 (2024-02-28)
### 🐛 Bugs Fixed
- [#2407](https://github.com/Azure/azureml-assets/pull/2407) Surface properties in the spec configuration

## 1.16.34 (2024-01-29)
### 🐛 Bugs Fixed
- [#2196](https://github.com/Azure/azureml-assets/pull/2196) Allow evaluation results to have names similar to models

## 1.16.33 (2024-01-22)
### 🐛 Bugs Fixed
- [#2161](https://github.com/Azure/azureml-assets/pull/2161) Fix credential not found issue for asset validation

## 1.16.32 (2024-01-19)
### 🐛 Bugs Fixed
- [#2155](https://github.com/Azure/azureml-assets/pull/2155) Fix no-op model metadata update

## 1.16.31 (2024-01-19)
### 🚀 New Features
- [#2078](https://github.com/Azure/azureml-assets/pull/2078) Model spec minimum SKU validation

## 1.16.30 (2024-01-18)
### 🚀 New Features
- [#2141](https://github.com/Azure/azureml-assets/pull/2141) Support archiving models

## 1.16.29 (2023-01-17)
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

