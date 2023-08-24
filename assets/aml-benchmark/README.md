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

## Note
- All custom exceptions must be defined in `components/src/utils/exceptions.py`.
- All custom errors must be defined in `components/src/utils/error_definitions.py`.
- All custom error strings which can be shared among components must be defined in `components/src/utils/error_strings.py`.

# Before creating a PR, please make sure to go through the following points:
> These are not mandatory to run manually but recommended. Allows to detect issues early, which otherwise would be detected by failing workflows after PR creation.

## 1. Run tests
* create a `config.json` with your workspace details at the root of this repository. The contents of this file has the following template-
```
{
    "subscription_id": "<subscription-id>",
    "resource_group": "<resource-group-name>",
    "workspace_name": "<workspace-name>"
}
```
* From the root of this repo, run `pip install -r assets/aml-benchmark/requirements.txt` to install the dependencies.
* Run `pytest assets/aml-benchmark/tests -n <no_of_workers>` to run the tests.

## 2. Run code health check
In the root of the repo, run the following in **powershell**:
```
python scripts/validation/code_health.py -i assets/aml-benchmark/
```

## 3. Run copyright validation check
In the root of the repo, run the following in **powershell**:
```
python scripts/validation/copyright_validation.py -i assets/aml-benchmark/
```

## 4. Run doc style check
In the root of the repo, run the following in **powershell**:
```
python scripts/validation/doc_style.py -i assets/aml-benchmark/
```