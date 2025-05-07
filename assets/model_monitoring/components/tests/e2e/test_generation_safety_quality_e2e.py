# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the data drift model monitor component."""
import os
import shutil
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


def download_and_flatten_logs(ml_client, pipeline_job_name, step_name, download_dir="./log"):
    found = False
    for child in ml_client.jobs.list(parent_job_name=pipeline_job_name):
        if child.display_name == step_name:
            found = True
            print(f"Found step: {child.display_name} ({child.name})，开始下载日志...")
            # 下载到临时目录
            temp_dir = os.path.join(download_dir, "temp_" + child.name)
            ml_client.jobs.download(child.name, download_path=temp_dir, all=True)
            print(f"日志已下载到 {os.path.abspath(temp_dir)}")
            
            # 创建目标 log 目录
            os.makedirs(download_dir, exist_ok=True)
            count = 0
            
            # 遍历 temp_dir 下所有文件，复制到 download_dir 下（文件名加上 step/job 前缀防止重名）
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    src_path = os.path.join(root, file)
                    # 生成新文件名，防止同名覆盖
                    rel_path = os.path.relpath(src_path, start=temp_dir)
                    new_file_name = f"{child.name.replace('/', '_')}_{rel_path.replace('/', '_')}"
                    dst_path = os.path.join(download_dir, new_file_name)
                    shutil.copy2(src_path, dst_path)
                    count += 1
            print(f"已整理 {count} 个日志文件到 {os.path.abspath(download_dir)}")
            
            # 删除临时目录
            shutil.rmtree(temp_dir)
            
            # 列出所有日志文件，并预览内容
            print("\n----- log 目录下的所有日志文件 -----")
            for file in os.listdir(download_dir):
                file_path = os.path.join(download_dir, file)
                if os.path.isfile(file_path):
                    print(file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read(2000)
                            print(f"--- {file} 内容预览 ---")
                            print(content if content else "(文件为空)")
                            print("文件完毕\n\n\n\n")
                    except Exception as e:
                        print(f"无法读取 {file}: {e}")
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
            download_and_flatten_logs(
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
            download_and_flatten_logs(
                ml_client,
                pipeline_job.name,
                "generation_safety_quality_signal_monitor_output",
                download_dir="./logs"
            )

        assert pipeline_job.status == "Completed"