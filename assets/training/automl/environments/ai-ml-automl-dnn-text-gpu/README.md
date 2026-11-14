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
