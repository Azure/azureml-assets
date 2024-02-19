# Introduction 
`aml-benchmark` is the directory for benchmarking LLM models.

# Getting Started
Inside the `aml-benchmark` directory, there are two subdirectories:
1. `components`: Contains all of the components code required by pipelines.
2. `scripts`: Contains all of the scripts that extend component's functionality.
3. `test`: Contains all the tests for testing components - unit tests as well as e2e tests.

# Contributing
* `components/`
    * `src/`
        * `<component_name>/`: contains the source code for the component.
        * `utils/`: contains the code that is shared among components.
    * `<component_name>/`
        * `asset.yaml`: contains the asset definition.
        * `spec.yaml`: contains the component definition.
* `scripts/`
    * `data_loaders/`: contains the scripts for loading data.
* `tests/`
    * `data/`: contains the data required for the tests.
    * `pipelines/`: contains the pipelines to test the components, each component has its corresponding pipeline file.
    * `test_component_name.py`- contains the tests for the component, each component has its corresponding test file.

# Guidelines
- All custom exceptions must be defined in `components/src/utils/exceptions.py`.
- All custom errors must be defined in `components/src/utils/error_definitions.py`.
- All custom error strings which can be shared among components must be defined in `components/src/utils/error_strings.py`.
- import statements must follow the following order:
    - Standard library imports, followed by a newline.
    - Third party imports, followed by a newline.
    - Local application imports.
- Entry function for every component must satisfy the following criteria:
    - Must be defined in a script named `main.py`.
    - Must be named `main`.
    - Must be decorated with `swallow_all_exceptions`.
    - Must state all the arguments that the function accepts instead of accepting a single argument `argparse.Namespace`.
- Every test file for a component must have the following classes:
    - `Test<component_name>Component`: **Required**, contains the e2e tests for the component that requires submission to AML workspace. Try to keep the number of tests to a minimum while making sure all of the inputs are tested once.
        - All the tests inside this class must use a single experiment name i.e. `<component-name>-test` and each test can use the method name as the run's display name.
    - `Test<component_name>Script`: **Required**, contains the remaining e2e tests for the component. Must test all possible input combinations and the exceptions that the component can raise.
    - `Test<component_name>Unit`: **Optional**, contains the unit tests for the component.

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

# Release checklist

## 1. Component release
- From the root of this repo, you can run either of the following to install the dependencies:
    - `pip install -r assets/aml-benchmark/requirements.txt`
    - `conda env create -f assets/aml-benchmark/dev_conda_env.yaml`
- We need to make sure that the spec file is updated for all the components before kicking off the release process. From the root of this repo, run the following command to upgrade the components:
    ```
    python assets/aml-benchmark/scripts/_internal/upgrade_components.py [--env_version <version>]
    ```
    parameter `env_version` can take the following values:
    | **Value** | **Description** |
    | --- | --- |
    | `"latest"` | This is the default value. It will upgrade the components' environment to the latest version. |
    | `""` | This will keep the components' environment version as is. |
    | `"<specific_version>"` | This will upgrade the components' environment to the specified version. |