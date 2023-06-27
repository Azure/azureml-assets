# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import datasets

import os
import pandas as pd
from sklearn import preprocessing
from azure.ai.ml.entities import AmlCompute, PipelineJob
from azure.ai.ml import dsl, load_component, MLClient
from azure.identity import DefaultAzureCredential
import time

try:
    from azureml.core import Workspace
    azureml_core_installed = True
except ImportError:
    print("azureml.core not found")
    azureml_core_installed = False


NUM_TEST_SAMPLES = 100


def load_dataset(split):
    dataset = datasets.load_dataset("DeveloperOats/DBPedia_Classes", split=split)
    return pd.DataFrame({"text": dataset["text"], "l1": dataset["l1"]})


def transform_dataset(le, dataset):
    dataset["label"] = le.transform(dataset["l1"])
    dataset = dataset.drop(columns="l1")
    return dataset


def fetch_and_write_dbpedia_dataset():
    train_path = "./resources/dbpedia_train/"
    test_path = "./resources/dbpedia_test/"

    pd_data = load_dataset("train")
    pd_test_data = load_dataset("test")
    # encode the labels
    encoded_classes = ['Agent', 'Device', 'Event', 'Place', 'Species',
                       'SportsSeason', 'TopicalConcept', 'UnitOfWork',
                       'Work']
    le = preprocessing.LabelEncoder()
    le.fit(encoded_classes)

    pd_data = transform_dataset(le, pd_data)
    pd_test_data = transform_dataset(le, pd_test_data)

    train_data = pd_data[NUM_TEST_SAMPLES:]
    test_data = pd_test_data[:NUM_TEST_SAMPLES]

    # Add some known error instances to make the data more interesting
    error_indices = [101, 319, 391, 414, 894, 1078, 1209]
    error_data = pd_test_data.iloc[error_indices]
    test_data = test_data.append(error_data)
    test_data = test_data.reset_index(drop=True)

    print("Saving to files")
    train_data.to_parquet(os.path.join(train_path, "dbpedia_train.parquet"), index=False)
    test_data.to_parquet(os.path.join(test_path, "dbpedia_test.parquet"), index=False)


def submit_and_wait(ml_client, pipeline_job) -> PipelineJob:
    created_job = ml_client.jobs.create_or_update(pipeline_job)
    assert created_job is not None

    while created_job.status not in ['Completed', 'Failed', 'Canceled', 'NotResponding']:
        time.sleep(30)
        created_job = ml_client.jobs.get(created_job.name)
        print("Latest status : {0}".format(created_job.status))
    assert created_job.status == 'Completed'
    return created_job


def submit_training_job():
    credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
    try:
        ml_client = MLClient.from_config(credential=credential,
                                         logging_enable=True)
    except Exception:
        if not azureml_core_installed:
            print("failed to load azureml.core and mlclient")
            raise
        # In case of exception due to config missing, try to get and create config, which may work if in compute instance
        workspace = Workspace.from_config()
        workspace.write_config()
        ml_client = MLClient.from_config(credential=credential,
                                         logging_enable=True)
    yaml_filename = "fetch_dbpedia_model.yml"
    fetch_model_component = load_component(
        source=yaml_filename
    )

    compute_name = "cpu-cluster"
    all_compute_names = [x.name for x in ml_client.compute.list()]

    if compute_name in all_compute_names:
        print(f"Found existing compute: {compute_name}")
    else:
        my_compute = AmlCompute(
            name=compute_name,
            size="STANDARD_DS3_V2",
            min_instances=0,
            max_instances=4,
            idle_time_before_scale_down=3600
        )
        ml_client.compute.begin_create_or_update(my_compute)
        print("Initiated compute creation")

    ml_client.components.create_or_update(fetch_model_component)

    model_base_name = "dbpedia_model"
    model_name_suffix = "1"
    device = "-1"

    @dsl.pipeline(
        compute=compute_name,
        description="Fetch DBPedia Model",
        experiment_name=f"Fetch_DBPedia_Model_{model_name_suffix}",
    )
    def my_training_pipeline(model_base_name, model_name_suffix, device):
        trained_model = fetch_model_component(
            model_base_name=model_base_name,
            model_name_suffix=model_name_suffix,
            device=device
        )
        trained_model.set_limits(timeout=120)

        return {}
    model_registration_pipeline_job = my_training_pipeline(
        model_base_name, model_name_suffix, device)

    # This is the actual submission
    submit_and_wait(ml_client, model_registration_pipeline_job)


fetch_and_write_dbpedia_dataset()
submit_training_job()
