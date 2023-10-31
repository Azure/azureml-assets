## Context
Model management unit tests are run as part of GitHub PR workflow which is defined here: [training-model-mgmt-unittests.yaml](https://github.com/Azure/azureml-assets/blob/main/.github/workflows/training-model-mgmt-unittests.yaml). 

This gets executed for following paths in the repository:

```
- assets/training/model_management/**
- assets/common/components/register_model/**
- assets/common/src/model_registration/**
- assets/common/components/mlflow_model_local_validation/**
- assets/common/src/mlflow_model_local_validation/**
- .github/workflows/training-model-mgmt-unittests.yaml
```


## Setup locally
Below steps should be executed with project directory as current directory.


```bash
# change cwd
cd assets/training/model_management

# create conda env
conda env create -f tests/dev_conda_env.yaml -q
conda activate model_mgmt

# copy src/azureml to tests/
cp -r src/azureml/ tests/unittests

# Run tests on bash shell
python -m pytest tests/unittests --tb=native --junitxml=$pytest_reports/test-result.xml -x -n 1 -ra --show-capture=no
```

### Run tests in VSCode

1. Complete all the steps till copying src/azureml to tests/unittests above
2. Create a vscode settings file if not already under project/.vscode/settings.json
3. Update settings.json with below params:
```json
{
    "python.formatting.provider": "black",
    "python.testing.pytestArgs": [
        "assets/training/model_management"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true
}
```
4. You should be able to see tests under tests in vscode. Fix errors if any by checking VSCode Python logs under Output
