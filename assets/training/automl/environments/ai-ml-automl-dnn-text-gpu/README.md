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
