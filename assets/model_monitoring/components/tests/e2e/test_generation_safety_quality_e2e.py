# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the data drift model monitor component."""
import os
import pytest
from azure.ai.ml import MLClient, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.exceptions import JobException
from tests.e2e.utils.constants import (
    COMPONENT_NAME_GENERATION_SAFETY_QUALITY_SIGNAL_MONITOR,
    DATA_ASSET_GROUNDEDNESS_PREPROCESSED_TARGET_DATA,
    DATA_ASSET_TRACE_LOGS_DATA_WITH_CONTEXT,
)


def _submit_generation_safety_quality_model_monitor_job(
    submit_pipeline_job, ml_client: MLClient, get_component, experiment_name,
    production_data, data_columns_dict: dict = {}, expect_failure: bool = False
):
    generation_safety_quality_signal_monitor = get_component(
        COMPONENT_NAME_GENERATION_SAFETY_QUALITY_SIGNAL_MONITOR
    )

    metrics = [
        "AcceptableCoherenceScorePerInstance",
        "AggregatedCoherencePassRate",
        "AcceptableFluencyScorePerInstance",
        "AggregatedFluencyPassRate",
        "AggregatedGroundednessPassRate",
        "AcceptableGroundednessScorePerInstance",
        "AggregatedRelevancePassRate",
        "AcceptableRelevanceScorePerInstance"
    ]
    metric_names = ",".join(metrics)
    prompt_column_name = data_columns_dict.get('prompt_column_name', None)
    completion_column_name = data_columns_dict.get('completion_column_name', None)
    ground_truth_column_name = data_columns_dict.get('ground_truth_column_name', None)
    context_column_name = data_columns_dict.get('context_column_name', None)

    @pipeline()
    def _generation_safety_quality_signal_monitor_e2e():
        generation_safety_quality_signal_monitor_output = generation_safety_quality_signal_monitor(
            signal_name="my-gsq-signal",
            monitor_name="my-gsq-model-monitor",
            production_data=production_data,
            metric_names=metric_names,
            model_deployment_name="gpt-4",
            sample_rate=1,
            monitor_current_time="2023-02-02T00:00:00Z",
            workspace_connection_arm_id="test_connection",
            prompt_column_name=prompt_column_name,
            completion_column_name=completion_column_name,
            ground_truth_column_name=ground_truth_column_name,
            context_column_name=context_column_name,
        )
        return {
            "signal_output": generation_safety_quality_signal_monitor_output.outputs.signal_output
        }

    pipeline_job = _generation_safety_quality_signal_monitor_e2e()
    pipeline_job.outputs.signal_output = Output(type="uri_folder", mode="direct")

    pipeline_job = submit_pipeline_job(
        pipeline_job, experiment_name, expect_failure
    )

    # Wait until the job completes
    try:
        ml_client.jobs.stream(pipeline_job.name)
    except JobException:
        # ignore JobException to return job final status
        pass

    return ml_client.jobs.get(pipeline_job.name)


def download_step_logs(ml_client, pipeline_job_name, step_name, download_dir="./logs"):
    """
    查找指定 pipeline job 下的 step，并下载日志到本地，并列出所有日志文件。
    """
    found = False
    for child in ml_client.jobs.list(parent_job_name=pipeline_job_name):
        if child.display_name == step_name or child.name.endswith(step_name):
            found = True
            print(f"Found step: {child.display_name} ({child.name})，开始下载日志...")
            ml_client.jobs.download(child.name, download_path=download_dir, all=True)
            step_dir = os.path.join(download_dir, child.name)
            print(f"日志已下载到 {os.path.abspath(step_dir)}")
            
            # 列出 logs 目录下的所有文件
            logs_dir = os.path.join(step_dir, 'logs')
            if os.path.exists(logs_dir):
                print("\n----- logs 目录下的所有日志文件 -----")
                for root, dirs, files in os.walk(logs_dir):
                    for file in files:
                        log_path = os.path.join(root, file)
                        rel_path = os.path.relpath(log_path, start=logs_dir)
                        print(rel_path)
                        # 可选：打印每个日志文件的前200字符
                        try:
                            with open(log_path, 'r', encoding='utf-8') as f:
                                print(f"--- {rel_path} 内容预览 ---")
                                print(f.read(2000))
                                print("文件完毕\n\n\n\n")
                        except Exception as e:
                            print(f"无法读取 {rel_path}: {e}")
            else:
                print(f"未找到 logs 目录: {logs_dir}")
            break
    if not found:
        print(f"未找到 step: {step_name}，请确认名称是否正确。")

@pytest.mark.gsq_test
@pytest.mark.e2e
class TestGenerationSafetyQualityModelMonitor:
    """Test class for GSQ."""

    def test_generation_safety_quality_successful(
        self, ml_client: MLClient, get_component, submit_pipeline_job, test_suite_name
    ):
        """Test GSQ is successful with traditional expected data."""
        pipeline_job = _submit_generation_safety_quality_model_monitor_job(
            submit_pipeline_job,
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_GROUNDEDNESS_PREPROCESSED_TARGET_DATA,
            {
                "prompt_column_name": "question",
                "completion_column_name": "answer",
            }
        )

        job_details = ml_client.jobs.get(pipeline_job.name)
        if pipeline_job.status != "Completed":
            print(job_details)
            # 自动下载 logs
            download_step_logs(
                ml_client,
                pipeline_job.name,
                "generation_safety_quality_signal_monitor_output",
                download_dir="./logs"
            )

        assert pipeline_job.status == "Completed"

    def test_generation_safety_quality_genai_successful(
        self, ml_client: MLClient, get_component, submit_pipeline_job, test_suite_name
    ):
        """Test GSQ is successful with genai trace logs."""
        pipeline_job = _submit_generation_safety_quality_model_monitor_job(
            submit_pipeline_job,
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_TRACE_LOGS_DATA_WITH_CONTEXT,
            {
                "prompt_column_name": "question",
                "completion_column_name": "output",
                "context_column_name": "context",
            }
        )

        job_details = ml_client.jobs.get(pipeline_job.name)
        if pipeline_job.status != "Completed":
            print(job_details)
            # 自动下载 logs
            download_step_logs(
                ml_client,
                pipeline_job.name,
                "generation_safety_quality_signal_monitor_output",
                download_dir="./logs"
            )

        assert pipeline_job.status == "Completed"