# Scikit-learn 1.9 environment

This environment provides Scikit-learn 1.9 and the latest Intel Extension for
Scikit-learn on the MCR OpenMPI 5.0 Ubuntu 24.04 base.
The workload interpreter is Python 3.11 in `/azureml-envs/sklearn-1.9`; changing it
does not require replacing the base image or its interpreter.

The Dockerfile creates the Conda environment before installing pip packages to
limit build-agent memory use. The Conda YAML uses the same package versions.
Pip is bumped to 26.2.2, which is available on both PyPI and conda-forge.
Azure ML packages and Intel Extension retain their latest-version macros, and
security requirements are preserved from `sklearn-1.5`.

The existing `sklearn-1.5` asset is unchanged. This separate asset upgrades
Scikit-learn instead of downgrading Intel Extension to support older sklearn APIs.

## Tests

`tests/sklearn_sample_test.py` builds the image and runs the diabetes regression
sample in standard and Intel modes using uniquely versioned test environments.
The sample checks Python/sklearn versions, finite predictions, model reload
parity, and that Intel mode constructs the patched estimator.

Set `subscription_id`, `resource_group`, `workspace`, and `cpu_cluster` before
running the test. Image macros must first be rendered as described in the
[environment build guide](https://github.com/Azure/azureml-assets/wiki/Environments#testing-environment-image-builds).
