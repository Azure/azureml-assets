# Overview
This repo is the home for built-in registry assets for Azure Machine Learning. The term "assets" currently includes [components](https://docs.microsoft.com/en-us/azure/machine-learning/concept-component) and [environments](https://docs.microsoft.com/en-us/azure/machine-learning/concept-environments). For more information on registries, please refer to [their documentation](https://github.com/Azure/azureml-previews/tree/main/previews/registries).

# Life Cycle
Once an asset is in a public registry, all Azure ML customers will have access to it and removing it would be difficult. The following asset life cycle stages are recommended as assets progress toward General Availability, and beyond:

| Stage | Stability | Availability | Display Name Suffix | Version Format | Notes |
|-------|-----------|--------------|---------------------|----------------|-------|
| Under Development | None | Source code is in repo, being developed against your own workspace/registry. Not in  public registry. | | | |
| Private Preview | In early testing | In public registry | (Private Preview) | #.#.#-preview | By end of Private Preview, asset *could be deleted* if not moving forward. |
| Public Preview | Relatively stable | In public registry |(Public Preview) | #.#.#-preview | By end of Public Preview, asset *can't be deleted*, but can be archived (see below). |
| General Availability | Stable, will have long-term support | In public registry | | #.#.# | **No interface breaking changes are allowed**. If there are breaking changes, create another asset. |
| Archived | No longer supported | In public registry, but unlisted | (Archived) | | |

**Display Name Suffix** in the table above is applied to an asset's display name and relates to its life cycle stage. For example, an asset with a display name of "My Component" that's reached the Public Preview stage would be updated to "My Component<b> (Public Preview)</b>".

**Version Format** relates to an asset's version. For example, an asset with a version of 1.0.0 that's in either of the preview stages would be updated to a version of 1.0.0<b>-preview</b>. After it reaches GA the **-preview** suffix would be dropped.

# Adding New Assets
Here are a some notes to help you get started:
* First create the asset in your workspace. The asset schema is the same, whether it's in a workspace or registry.
* [Get familliar with registries](https://github.com/Azure/azureml-previews/tree/main/previews/registries). Develop the asset in your own registry and get it working there first. 
* Here is an example component: https://github.com/Azure/azureml-assets/tree/main/training/vision/components/object_detection
* Information on creating environments is in the [wiki](https://github.com/Azure/azureml-assets/wiki/Environments)
* Folder structure:
  * `/<area>/<sub area>/<assset type>/<asset name>`
  * Example: `/training/vision/components/object_detection`
* Doc/Sample
  * Add a "Learn More" link to the description, for example
  ```description: Trains an object detection model. [Learn More](https://aka.ms/built-in-vision-components)```
  * Use aka.ms links. Before this repo is public, you could create your sample in [azureml-examples](https://github.com/Azure/azureml-examples/tree/main).
  * We will create official docs in https://docs.microsoft.com later.
* Once everything is ready, create an [asset.yaml](https://github.com/Azure/azureml-assets/blob/release/latest/component/train_object_detection_model/object_detection/asset.yaml) file.
* Test jobs
  * Create test jobs, which will be run before the asset is published to production.
  * You will need to have a `tests.yml` inside your folder to indicate which files will be involved during the test and you may also indicate any pre or post scripts that need to be executed. More information about `tests.yml` can be found in this repo's [wiki](https://github.com/Azure/azureml-assets/wiki/Adding-Test-Job).
  * For now, we ask you to use asset IDs to reference components in the YAML files of testing jobs. In the future, we will allow you to use local references.
  * An asset ID follow this format:
  `azureml://registries/azureml-dev/{asset_type}/{asset_name}/version/{version}`. Please ensure `asset_name` and `version` match the values in your `spec.yml` file, otherwise the test jobs will fail to find the asset.
* Your asset will first be published to azureml-dev and azureml-staging. You could test the component from those two registries first.
* Once everything is ready, submit a PR to this repo. We will help you publish the component.

# Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

# Data Collection
The software may collect information about you and your use of the software and send it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry as described in the repository. There are also some features in the software that may enable you and Microsoft to collect data from users of your applications. If you use these features, you must comply with applicable law, including providing appropriate notices to users of your applications together with a copy of Microsoft’s privacy statement. Our privacy statement is located at https://go.microsoft.com/fwlink/?LinkID=824704. You can learn more about data collection and use in the help documentation and our privacy statement. Your use of the software operates as your consent to these practices.

Information on managing Azure telemetry is available at https://azure.microsoft.com/en-us/privacy-data-management/.
