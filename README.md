# Overview
This repo is the home for built-in registry assets from Azure Machine Learning Service. Assets includes Components, Environments for now. For more information on registry, please see [here](https://github.com/Azure/azureml-previews/tree/main/previews/registries)

# Life Cycle
There's no hidden-based-on-asset mechanism right now. So once the asset is in built-in registry, all users will have access, removing it is hard. It's understandable that assets have different stability levels. So we recommend following this asset lifecycle stages:
* **Under Development**: 
  * Source code is in repo, development against your own workspace/registry.
  * Not published in builtin registry, end user can't see.
* **Private Preview**: 
  * Asset is published to built-in registry, but still in early testing.
  * To indicate an asset is in Private Preview, we suggest:
    * Display Name: Suffix with **(Private Preview)**
    * Version: x.x.x-preview
  * By end of Private Preview, asset **could be deleted** if we don't want to move forward
* **Public Preview**:
  * Asset is relatively stable. 
  * To indicate an asset is in Public Preview, we suggest:
    * Display Name: Suffix with **(Public Preview)**
    * Version: x.x.x-preview
  * By end of Public Preview, asset **can't be deleted**, but can be archived. You can still continue to reference and use an archived asset, but it will be hidden in the list.
* **GA**: 
  * Asset is stable and we intend to support in the long term
  * To indicate an asset is in GA, we suggest:
    * Display Name: No additional suffix
    * Version: x.x.x (no additional suffix anymore)
  * **No-interface breaking changes are allowed**. If there are breaking changes, please another asset.
* **Archived**:
  * Asset isn't being supported in the long term anymore, but user can still use it
  * To indicate an asset is in GA, we suggest:
    * Display Name: Suffix with **(Archived}**
  * Archived asset can still be used, but is hidden from the list

# Adding New Assets
Here's the list of things you do when you are getting started:
* You can also first target to create the asset in your workspace, the asset schema in registry and workspace are exactly the same.
* (Understand and get familliar with Registry)[https://github.com/Azure/azureml-previews/tree/main/previews/registries]. Developement the asset in your own registry and make sure it's working first. 
* Here is an example component: https://github.com/Azure/azureml-assets/tree/main/training/vision/components/object_detection
* Folder structure:
  * ```/<area>/<sub area>/<assset type>/<asset>```
  * e.g. ```/training/vision/components/object_detection```
* Doc/Sample
  * Add "Learn More" link to description, for example
  ```description: Trains an object detection model. [Learn More](https://aka.ms/built-in-vision-components)```
  * Use aka.ms links, before this repo is public, you could create your sample in [azureml-examples](https://github.com/Azure/azureml-examples/tree/main).
  * We will create official docs in doc.microsoft.com later.
* Once everything is ready, create [asset.yaml](https://github.com/Azure/azureml-assets/blob/release/latest/component/train_object_detection_model/object_detection/asset.yaml) file.
* Test jobs
  * Provide test jobs, these will be running before we publish the asset to production.
  * You will need to have a `tests.yml` inside your folder to indicate which files will be involved during the test and you may also indicate any pre or post scripts need to be executed during the whole process of testing. The details of `tests.yml` requirement can be found in [Wiki](https://github.com/Azure/azureml-assets/wiki/Adding-Test-Job) of this repository.
  * For now, we ask you to use **Asset ID** for components referencing in the yml files of testing jobs. In the future, we will allow you to use local references.
  * An example of Asset ID:
  `azureml://registries/azureml-dev/{asset_type}/{asset_name}/version/{version}`
  please double check the `asset_name` and `version` are the same as what indicated in your `spec.yml`. Otherwise, the test jobs will definitely fail due to the null referencing.
* Your asset will first be published to azureml-dev and azureml-staging. You could test the component from those two registries first.
* Once everything is ready, submit a review request in this repo. We will help you publish the component.

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
