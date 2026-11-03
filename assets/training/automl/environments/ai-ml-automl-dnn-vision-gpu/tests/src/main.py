# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Smoke tests for vision dependencies, YOLO ONNX export, and sklearn training."""
# imports
import importlib
import mlflow
import argparse
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# define functions
def main(args):
    """Run and evaluate model."""
    validate_vision_dependencies()
    validate_yolo_onnx_export()
    # enable auto logging
    mlflow.autolog()
    # setup parameters
    params = {
        "fit_intercept": args.fit_intercept,
        "positive": args.positive,
    }
    # read in data
    df = pd.read_csv(args.diabetes_csv)

    # process data
    X_train, X_test, y_train, y_test = process_data(df, args.random_state)

    # train model
    model = train_model(params, X_train, X_test, y_train, y_test)

    print(model)


def validate_vision_dependencies():
    """Check the AutoML import chain and compiled torchvision operators."""
    import torch
    from torchvision.ops import nms

    importlib.import_module("azureml.automl.dnn.vision.object_detection.runner")
    assert torch.cuda.is_available(), "CUDA is not available in the GPU test environment"
    for device in ("cpu", "cuda"):
        boxes = torch.tensor(
            [[0, 0, 2, 2], [0, 0, 2, 2], [4, 4, 6, 6]],
            dtype=torch.float32,
            device=device,
        )
        scores = torch.tensor([0.9, 0.8, 0.7], device=device)
        assert nms(boxes, scores, 0.5).tolist() == [0, 2]
        print(f"torchvision NMS passed on {device}")


def validate_yolo_onnx_export():
    """Export real YOLO graphs and verify legacy batch and normalization behavior."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from azureml.automl.dnn.vision.object_detection_yolo.models.yolo import Model
    from azureml.automl.dnn.vision.object_detection_yolo.models.yolo_wrapper import (
        YoloV5Wrapper,
    )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        config, _ = YoloV5Wrapper._find_config("yolov5", "small")
        model = Model(model_cfg=config, nc=2).eval()
        model.hyp = {"img_size": 64}
        wrapper = SimpleNamespace(model=model, distributed=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        with TemporaryDirectory() as directory:
            for enable_norm in (False, True):
                path = Path(directory) / f"yolo-normalized-{enable_norm}.onnx"
                YoloV5Wrapper.export_onnx_model(
                    wrapper, file_path=str(path), device="cpu", enable_norm=enable_norm
                )
                graph = onnx.load(str(path))
                onnx.checker.check_model(graph)
                assert next(op.version for op in graph.opset_import if not op.domain) == 11
                assert graph.graph.input[0].name == "input"
                assert graph.graph.output[0].name == "output"
                for value in (graph.graph.input[0], graph.graph.output[0]):
                    assert value.type.tensor_type.shape.dim[0].dim_param == "batch"
                session = ort.InferenceSession(
                    str(path), sess_options=options, providers=["CPUExecutionProvider"]
                )
                for batch_size in (1, 2):
                    inputs = torch.rand(batch_size, 3, 64, 64)
                    raw_inputs = inputs * 255 if enable_norm else inputs
                    with torch.no_grad():
                        expected = model(raw_inputs / 255 if enable_norm else raw_inputs)[0]
                    actual = session.run(["output"], {"input": raw_inputs.numpy()})[0]
                    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-4)
                print(f"YOLO legacy ONNX export passed: normalized={enable_norm}, batches=1,2")
    finally:
        torch.set_num_threads(previous_threads)


def process_data(df, random_state):
    """Process data."""
    # split dataframe into X and y
    X = df.drop(["target"], axis=1)
    y = df["target"]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # return splits and encoder
    return X_train, X_test, y_train, y_test


def train_model(params, X_train, X_test, y_train, y_test):
    """Train the model."""
    # train model
    model = LinearRegression(**params)
    model = model.fit(X_train, y_train)

    # return model
    return model


def parse_args():
    """Parse arguments."""
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--diabetes-csv", type=str)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--fit_intercept", type=bool, default=True)
    parser.add_argument("--positive", type=bool, default=False)
    parser.add_argument("--intel-extension", type=bool, default=False)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()
    if (args.intel_extension):
        from sklearnex import patch_sklearn
        patch_sklearn()
    # run main function
    main(args)
