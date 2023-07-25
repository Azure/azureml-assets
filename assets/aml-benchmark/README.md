# Introduction 
`aml-benchmark` is the directory for benchmarking LLM models.

# Getting Started
Inside the `aml-benchmark` directory, there are two subdirectories-
1. `components`: Contains all of the components code required by pipelines.
2. `test`: Contains all the tests for testing components - unit tests as well as e2e tests.

# Contributing
* `components/`
    * `src/`
        * `<component_name>/`: contains the source code for the component.
        * `utils/`: contains the code that is shared among components.
    * `<component_name>/`
        * `asset.yaml`: contains the asset definition.
        * `spec.yaml`: contains the component definition.
* `tests/`
    * `data/`: contains the data required for the tests.
    * `pipelines/`: contains the pipelines to test the components, each component has its corresponding pipeline file.
    * `test_component_name.py`- contains the tests for the component, each component has its corresponding test file.

# Before creating a PR, please make sure to do the following:

## Run tests
* create a `config.json` with your workspace details at the root of this repository. The contents of this file has the following template-
```
{
    "subscription_id": "<subscription-id>",
    "resource_group": "<resource-group-name>",
    "workspace_name": "<workspace-name>"
}
```
* From the root of this repo, Run `pip install -r assets/aml-benchmark/requirements.txt` to install the dependencies.
* Run `cd assets/aml-benchmark/components/src` to change the directory, current directory must be set to **assets/aml-benchmark/components/src**.
* Run `pytest ../../tests -n <no_of_workers>` to run the tests.

## Run code health check
In the root of the repo, run the following in **powershell**:
```
python scripts/validation/code_health.py -i assets/aml-benchmark/
```

## Run copyright validation check
In the root of the repo, run the following in **powershell**:
```
python scripts/validation/copyright_validation.py -i assets/aml-benchmark/
```
