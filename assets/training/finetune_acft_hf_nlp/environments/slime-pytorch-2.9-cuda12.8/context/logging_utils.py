import logging
import os

import wandb

from . import wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False
_MLFLOW_INITIALIZED = False
_MLFLOW_AVAILABLE = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _init_mlflow():
    """Initialize MLflow for Azure ML metric logging.

    AML sets MLFLOW_TRACKING_URI automatically; when present, metrics logged
    via mlflow.log_metrics() appear in the AML run's Metrics tab.
    """
    global _MLFLOW_INITIALIZED, _MLFLOW_AVAILABLE
    if _MLFLOW_INITIALIZED:
        return _MLFLOW_AVAILABLE
    _MLFLOW_INITIALIZED = True

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return False

    try:
        import mlflow

        mlflow.autolog(disable=True)
        _MLFLOW_AVAILABLE = True
        logging.getLogger(__name__).info(
            "MLflow tracking enabled (URI: %s)", os.environ["MLFLOW_TRACKING_URI"]
        )
    except ImportError:
        logging.getLogger(__name__).info("mlflow not installed, AML metric logging disabled")
    except Exception:
        logging.getLogger(__name__).exception("Failed to initialize mlflow")
    return _MLFLOW_AVAILABLE


def _log_mlflow(metrics: dict, step: int):
    """Log numeric metrics to MLflow / Azure ML."""
    try:
        import mlflow

        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if numeric:
            mlflow.log_metrics(numeric, step=step)
    except Exception:
        pass


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)

    _init_mlflow()


def update_tracking_open_metrics(args, router_addr):
    wandb_utils.reinit_wandb_primary_with_open_metrics(args, router_addr)


def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])

    if _MLFLOW_AVAILABLE:
        step = int(metrics.get(step_key, 0))
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _log_mlflow(metrics_except_step, step=step)
