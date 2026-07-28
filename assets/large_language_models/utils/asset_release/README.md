# Asset release

Azure Machine Learning (AML) relies on Azure DevOps to build and release curated environments and components. The scripts in this folder can be used to check, monitor and manage such release.

## list_asset_release.py

This script is used to list the information of last 10 releases.

### Prerequisite

Create a Personal Access Token (PAT) in Azure DevOps including at lease "Build (Read)" and "Release (Read)" access, if you haven't. [Using Personal Access Token](https://learn.microsoft.com/en-us/azure/devops/organizations/accounts/use-personal-access-tokens-to-authenticate?view=azure-devops&tabs=Windows#about-pats)

Set local environment:
`set AZURE_DEVOPS_PAT=<YOUR_PAT>`

Install required Python packages

`pip install azure-devops msrest`

### Usage

`python list_assets_release.py -p 'environment/llm'` -v

Above script will show the more details about last environment/llm build release.

## list_assets_build.py

This script is used to list the successful asset builds.

### Usage

`python list_assets_build.py -p 'llm-rag'`
Above command will show the last 3 build that with build pattern 'llm-rag'