# Introduction 
This doc talks about the pre-requisites for component release:

- From the root of this repo, run the following to install the dependencies: `conda env create -f assets/training/scripts/_component_upgrade/dev_conda_env.yaml`.
- From the root of this repo, run the following to upgrade the components: `python assets/training/scripts/_component_upgrade/main.py`.
- After successful completion, the script will:
    - print the regular expression, copy this since this wil be required in the build pipeline.
    - generate changes. Create a PR and merge your changes into main branch. Wait for 10 minutes after PR merge and then kick off the build pipeline.
