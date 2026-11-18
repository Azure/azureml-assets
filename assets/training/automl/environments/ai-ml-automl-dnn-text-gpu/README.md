# ai-ml-automl-dnn-text-gpu Docker Environment

## Overview
This environment is used by Azure ML AutoML for training models.
It is not intended for use in other scenarios and is subject to change without notice.

## ONNX compatibility

The ONNX security upgrade must be accompanied by compatible `skl2onnx`,
`onnxmltools`, and `onnxconverter-common` releases. Older converters import
`onnx.mapping`, which is no longer available, and prevent the AutoML NER runner
from starting. The image retains its patched ONNX floor and uses the converter
versions already validated in the AutoML vision image.

The shared AutoML runtime also needs the vision image's Python 3.11 ONNX guard
and `importlib.resources` backports to work with the upgraded converters and
patched setuptools. The smoke test imports the real NER runner and checks
scikit-learn-to-ONNX prediction parity before its existing training example.

The final datasets upgrade also resolves with `pyarrow==23.0.1` and the existing
NumPy 1.23 cap. This retains the patched Arrow release used by the vision image
without selecting newer Arrow wheels that require NumPy 2. The datasets
security floor remains `>=5.0.1`.

## NLP entrypoint imports

The shared training runtime eagerly imports tabular and distributed entrypoints
when text runners import its common logging utility. Those unused entrypoints
load Dask dataframe, whose patched releases require newer pandas and NumPy than
the AutoML runtime supports. The image backports lazy loading for the existing
entrypoint module exports, preserving their names and import errors when used.
NER, multiclass, and multilabel runners can load common logging without loading
unrelated forecasting dependencies. The pandas/NumPy pins and the patched
`distributed>=2026.1.0` floor remain unchanged.

## MLflow logging

Azure ML tracking does not support MLflow 3's logged-models API. Until the
runtime package includes the fix, the image backports legacy logging for
custom-loader models, using the current child run ID. MLflow 2 keeps its
existing logging path.
