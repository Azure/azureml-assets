# Overview
RFC (Request For Comment) folder is used for creating specs that's relatively big, such as new asset proposals etc.

# New Asset Proposal Request
If you know the owning area, organize this folder by <area name>/xxx.md
In the markdown, please create a new PR answering the following questions.

* What are the assets being proposed?
  Model Evaluation DPV2 component
  
  
•	Why should this asset be built-in?

Model Evaluation would be a part of DPV2 which allows user to run Model Evaluation for Any Machine Learning model on Azure ML.  In order for us to make the functionality of model evaluation available as generic functionality at Azure Machine Learning level, this component should be available as built-in componenet which users should be able to use it as drag and drop in designer.

Model Evaluation will be a standalone component available to user in all 3 user-experience. Currently, we have implemented the model evaluation feature as Command Component. SDK and CLI internally create a pipeline job which consumes model evaluation component. 3 user experiences as below: 

> V2 SDK : Model Evaluation Job can be created by user by consuming our component under a pipeline job. Our component can be a part of another pipeline or a standalone job as well. A sample code on how to consume our component using sdk is shown below:  

Sample Usage: run_sdk.py 
  
> Azure ML CLI  : Similarly, Model evaluation job can also be created using Azure ML CLI. A User has to create a Pipeline job YAML with component as ‘azureml:model_evaluation:<version>’ and specify all other input parameters including Test data which is passed as URI_FOLDER.   
  
 Sample YAML: model_evaluation_job.yml  
  
> Designer (UI)  : The model evaluation component will be available in Designer which allows user to drag and drop the component along with Test Data and filling in the rest of the component parameters and create a model evaluation job using UI.



•	Support model (what teams will be on the hook for bug fixes and security patches)?
PM (Sharmeelee Bijlani)
Dev Lead (Shipra Jain, Anup Shirgaonkar)

•	A high-level description of the implementation for each asset.

Model Evaluation would be a part of DPV2 which allows user to run Model Evaluation for Any Machine Learning model on Azure ML.  In order for us to make the functionality of model evaluation available as generic functionality at Azure Machine Learning level, this component should be available as built-in componenet which users should be able to use it as drag and drop in designer.