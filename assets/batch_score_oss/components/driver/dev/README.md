# Testing Locally

The [batch_score_simulator.py](https://msdata.visualstudio.com/Vienna/_git/batch-score?path=/driver/dev/batch_score_simulator.py) script can be used to test the driver script locally. It simulates a 1-instance, 1-process PRS scenario by partitioning the [MLTable](https://msdata.visualstudio.com/Vienna/_git/batch-score?path=/driver/dev/training) data into mini batches and execute the driver's `init()`, `run()`, and `shutdown()` functions appropriately, as well as write the results of `run()` to file.

This folder does not verify any yaml configurations.

## Create a virtual environment
- Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- Open Anaconda Prompt terminal (Search Anaconda in start menu).
- Create a new environment: `conda create -n batch_score_test_env python=3.8`
- Activate the environment: `conda activate batch_score_test_env`
- Ensure python version is 3.8.16: `python --version`

## Install dependencies
- Navigate to the root enlistment folder of the batch-score repo.
- Install the dependencies: `pip install -r driver\tests\requirements.txt`

## Create launch configuration file
- Navigate to the enlistment root. 
- Create a new folder `.vscode`.
- Create a new file `launch.json` with the following content:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Attach",
            "type": "python",
            "request": "launch",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/driver/dev",
            "program": "batch_score_simulator.py",
            "justMyCode": true,
            "env": {
                "PYTHONPATH": "${workspaceFolder}/driver"
            },
            "args": [
                "--debug_mode", "true",
                "--online_endpoint_url", "https://real-dv3-stable.centralus.inference.ml.azure.com/v1/engines/davinci/completions",
            ]
        }
    ]
}
```
## Running simulator.py
- Open Visual Studio Code from the conda environment terminal: `code .`
- Open command palette (Ctrl+Shift+P) and search for `Python: Select Interpreter`.
- Set the python interpreter to the python version of your conda environment.
- Hit `F5` to start a debugging session. 

### Optional simulator features

In addition to simulating the PRS runtime environment, the simulator script can also provide a simulated endpoint to score against, a simulated batch pool routing service, and a simulated quota/rate limiter service. Each of these can be used independently from the others, so you can pick and choose which dependencies are real and which are fake.

The simulated services are lenient with what inputs they accept. E.g., the quota simulator doesn't care what audience you request.

#### Endpoint simulator

To enable the simulated ML endpoint, provide the scoring URL `{{ENDPOINT_SIMULATOR_HOST}}/v1/engines/davinci/completions` either as the `--online_endpoint_url` command-line argument or as the routing simulator's endpoint when simulating a batch pool.

```json
"env": {
    "ENDPOINT_SIMULATOR_WORK_SECONDS": 10,
    // Set this if using the batch pool routing simulator (see next section).
    "ROUTING_SIMULATOR_ENDPOINT_URI": "{{ENDPOINT_SIMULATOR_HOST}}/v1/engines/davinci/completions",
},
"args": [
    // Set this if not using the batch pool feature.
    "--online_endpoint_url", "{{ENDPOINT_SIMULATOR_HOST}}/v1/engines/davinci/completions",
]
```

#### Routing simulator

To enable the simulated routing service, provide two values in your `launch.json` environment:

```json
"env": {
    "BATCH_SCORE_ROUTING_BASE_URL": "{{ROUTING_SIMULATOR_HOST}}/api",
    "ROUTING_SIMULATOR_ENDPOINT_URI": "{{ENDPOINT_SIMULATOR_HOST}}/v1/engines/ada/completions",
}
```

The `BATCH_SCORE_ROUTING_BASE_URL` variable tells the routing code in the client where find the fake routing service, and the `ROUTING_SIMULATOR_ENDPOINT_URI` variable tells the routing simulator itself what endpoint to return. You can set it to the endpoint simulator as in this example, or a real endpoint scoring URI. (The routing simulator will always return a single endpoint for any batch pool requested.)

#### Quota simulator

To enable the quota simulator, add the `BATCH_SCORE_QUOTA_BASE_URL` environment variable in your `launch.json`, and optionally also set `QUOTA_SIMULATOR_CAPACITY` to configure a specific amount of simulated total quota:

```json
"env": {
    "BATCH_SCORE_QUOTA_BASE_URL": "{{QUOTA_SIMULATOR_HOST}}/ratelimiter",
    "QUOTA_SIMULATOR_CAPACITY": "2048",
}
```

### Optional auth configuration

By default, running the batch score component locally will use your `az` login to fetch the access tokens it needs. If you wanted to override that token with a different one, you could either run `az login` to authenticate under a different account or manually write a token to a local file and pass it to the component via command line:

```json
"args": [
    "--token_file_path", "<Your Path Here>\\batch-score\\driver\\dev\\secrets\\token.txt"
]
```

### Command Line

Running simulator.py through command line is also possible.navigate to the `dev/` folder and run the script, passing in flags as needed. Again, ensure the appropriate Python version is used.

```bash
python simulator.py --debug_mode=True --online_endpoint_url=https://pr-wenbinmeng.eastus.inference.ml.azure.com/v1/engines/ada/completions --azureml_model_deployment=api-ci-ea27f087 --token_file_path=./secrets/token.txt
```

## Defining Data to Test against

Update the files in the `training/` folder with the data you would like to test.

Or create new datasets to use. An example of how to create an MLtable from Huggingface's CNN DailyMail data is documented in [create_dataset.py](./datasets/create_dataset.py)

## Full E2E Testing

First, create the component by navigating to the `yamls/components/` folder:

```bash
az ml component create --file dynamic_parallel_batch_score.yml
```

Then, create the job. Using the `quickstart/` directory is a good starting point. Refer to the quickstart [Pipeline Job Creation Step](../../quickstart/README.md#4-create-the-pipeline-job). The same readme shows instructions on how to configure your job through the CLI, as well as how to view output.

You can monitor the progress of the job through either ML Studio UI or the following CLI command:

```bash
az ml job show --name=<job name, a GUID>
```
