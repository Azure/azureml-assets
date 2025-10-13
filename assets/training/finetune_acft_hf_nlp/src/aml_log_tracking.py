# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
import os
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = ["wandb", "mlflow", "swanlab", "vemlp_wandb", "tensorboard", "console", "clearml", "trackio", "azureml"]

    def __init__(self, project_name, experiment_name, default_backend: str | list[str] = "console", config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb

            settings = None
            if config and config["trainer"].get("wandb_proxy", None):
                settings = wandb.Settings(https_proxy=config["trainer"]["wandb_proxy"])
            wandb.init(project=project_name, name=experiment_name, config=config, settings=settings)
            self.logger["wandb"] = wandb

        if "trackio" in default_backend:
            import trackio

            trackio.init(project=project_name, name=experiment_name, config=config)
            self.logger["trackio"] = trackio

        if "mlflow" in default_backend:
            import os

            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter(project_name, experiment_name)

        if "console" in default_backend:
            from verl.utils.logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

        if "azureml" in default_backend:
            self.logger["azureml"] = AzureMLLogger(project_name, experiment_name, config)

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def __del__(self):
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        if "vemlp_wandb" in self.logger:
            self.logger["vemlp_wandb"].finish(exit_code=0)
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()
        if "clearnml" in self.logger:
            self.logger["clearnml"].finish()
        if "trackio" in self.logger:
            self.logger["trackio"].finish()
        if "azureml" in self.logger:
            self.logger["azureml"].finish()


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, int | float | np.floating | np.integer):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This '
                    f"invocation of ClearML logger's function is incorrect so this attribute was dropped. "
                )

    def finish(self):
        self._task.mark_completed()


class AzureMLLogger:
    """
    AzureML logger adapter for VERL tracking system.
    Integrates with Azure Machine Learning Run context for metrics logging.
    """

    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.azureml_run = None
        self.logged_steps = []
        self.log_metrics_at_root = True
        self.set_log_prefix = True

        # Initialize AzureML run context
        try:
            from azureml.core.run import Run
            self.azureml_run = Run.get_context()

            # Check if running offline (local development)
            if self.azureml_run and "OfflineRun" in self.azureml_run.id:
                print("Warning: Running in offline mode - AzureML logging disabled")
                self.azureml_run = None
            else:
                print(f"AzureML logging initialized for experiment: {experiment_name}")

                # Log configuration as run properties if available
                if config:
                    try:
                        # Convert config to flat dict for run properties
                        flat_config = self._flatten_config(config)
                        for key, value in flat_config.items():
                            # AzureML run properties have string length limits
                            if isinstance(value, str) and len(value) < 1000:
                                self.azureml_run.tag(key, value)
                    except Exception as e:
                        print(f"Warning: Failed to log config as run properties: {e}")

        except ImportError:
            print("Warning: azureml-core not available - AzureML logging disabled")
            self.azureml_run = None
        except Exception as e:
            print(f"Warning: Failed to initialize AzureML run context: {e}")
            self.azureml_run = None

    def _flatten_config(self, config, parent_key='', sep='_'):
        """Flatten nested configuration dictionary"""
        items = []
        if isinstance(config, dict):
            for k, v in config.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(self._flatten_config(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, str(v)))
        return dict(items)

    def _should_log_to_parent(self):
        """Check if we should log to parent pipeline run"""
        if not self.azureml_run:
            return None

        try:
            parent_run = self.azureml_run.parent
            child_run = None

            # Navigate up the hierarchy to find pipeline run
            while parent_run is not None and (
                parent_run.type == "PipelineRun" or
                parent_run.type == "StepRun" or
                parent_run.type.lower() == "finetunerun"
            ):
                child_run = parent_run
                parent_run = parent_run.parent
            return child_run
        except Exception:
            return None

    def log(self, data, step):
        """Log metrics to AzureML run"""
        if not self.azureml_run:
            return

        try:
            from math import isnan

            for k, v in data.items():
                if isinstance(v, (int, float)) and not isnan(v) and (step, k) not in self.logged_steps:

                    # Handle prefix modification if needed
                    if not self.set_log_prefix:
                        eval_prefix = 'eval_'
                        train_prefix = 'train_'
                        if k.startswith(eval_prefix):
                            k = k[len(eval_prefix):]
                        if k.startswith(train_prefix):
                            k = k[len(train_prefix):]
                            k = k + '_train'

                    # Log to current run
                    self.azureml_run.log(k, v, description=k, step=step)

                    # Log to parent pipeline run if enabled
                    if self.log_metrics_at_root:
                        parent_run = self._should_log_to_parent()
                        if parent_run:
                            try:
                                parent_run.log(k, v, description=k, step=step)
                            except Exception as e:
                                print(f"Warning: Failed to log to parent run: {e}")

                    # Track logged steps to avoid duplicates
                    if step is not None:
                        self.logged_steps.append((step, k))

        except Exception as e:
            print(f"Warning: AzureML logging failed: {e}")

    def finish(self):
        """Cleanup method for AzureML logger"""
        # AzureML runs are managed by the platform, no explicit cleanup needed
        pass


class _TensorboardAdapter:
    def __init__(self, project_name, experiment_name):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{project_name}/{experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def log(self, data, step):
        import mlflow

        results = {k.replace("@", "_at_"): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    project_name: str = None
    experiment_name: str = None

    def log(self, loggers, samples, step):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

        if "vemlp_wandb" in loggers:
            self.log_generations_to_vemlp_wandb(samples, step)

        if "azureml" in loggers:
            self.log_generations_to_azureml(samples, step)

    def log_generations_to_vemlp_wandb(self, samples, step):
        from volcengine_ml_platform import wandb as vemlp_wandb

        self._log_generations_to_wandb(samples, step, vemlp_wandb)

    def log_generations_to_wandb(self, samples, step):
        import wandb

        self._log_generations_to_wandb(samples, step, wandb)

    def _log_generations_to_wandb(self, samples, step, wandb):
        """Log samples to wandb as a table"""

        # Create column names for all samples
        columns = ["step"] + sum(
            [[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], []
        )

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_table = swanlab.echarts.Table()

        # Create column names
        headers = ["step", "input", "output", "score"]

        swanlab_row_list = [[step, *sample] for sample in samples]
        swanlab_table.add(headers=headers, rows=swanlab_row_list)

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_table}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        table = [
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            # Use the same directory structure as _TensorboardAdapter
            if self.project_name and self.experiment_name:
                default_dir = os.path.join("tensorboard_log", self.project_name, self.experiment_name)
            else:
                default_dir = "tensorboard_log"

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", default_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            # Assuming sample contains [input, output, score]
            if len(sample) >= 3:
                input_text, output_text, score = sample[0], sample[1], sample[2]

                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)
        # Flush to ensure data is written
        self.writer.flush()

    def log_generations_to_azureml(self, samples, step):
        """Log validation generations to AzureML as metrics and artifacts"""
        try:
            from azureml.core.run import Run

            azureml_run = Run.get_context()
            if azureml_run and "OfflineRun" not in azureml_run.id:

                # Log summary metrics
                avg_score = sum(sample[2] for sample in samples if len(sample) >= 3) / len(samples)
                azureml_run.log("val/avg_generation_score", avg_score, step=step)
                azureml_run.log("val/num_generations", len(samples), step=step)

                # Log individual generation scores as a series
                for i, sample in enumerate(samples):
                    if len(sample) >= 3:
                        azureml_run.log(f"val/generation_{i+1}_score", sample[2], step=step)

                # Log generations as artifact (JSON format)
                import json
                import tempfile
                from pathlib import Path

                try:
                    generations_data = []
                    for i, sample in enumerate(samples):
                        generation_data = {
                            "step": step,
                            "generation_id": i + 1,
                        }

                        if len(sample) >= 3:
                            generation_data.update({
                                "input": sample[0],
                                "output": sample[1],
                                "score": sample[2]
                            })
                        else:
                            generation_data["data"] = sample

                        generations_data.append(generation_data)

                    # Create temporary file for the artifact
                    with tempfile.NamedTemporaryFile(mode='w', suffix=f'_val_generations_step_{step}.json', delete=False) as tmp_file:
                        json.dump(generations_data, tmp_file, indent=2)
                        tmp_file_path = tmp_file.name

                    # Upload as artifact to AzureML
                    azureml_run.upload_file(
                        name=f"val_generations/step_{step}_generations.json",
                        path_or_stream=tmp_file_path
                    )

                    # Clean up temporary file
                    Path(tmp_file_path).unlink()

                except Exception as e:
                    print(f"Warning: Failed to save validation generations artifact to AzureML: {e}")

        except ImportError:
            print("Warning: azureml-core not available for validation generations logging")
        except Exception as e:
            print(f"Warning: AzureML validation generations logging failed: {e}")
