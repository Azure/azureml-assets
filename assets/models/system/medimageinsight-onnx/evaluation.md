The model is optimized for inference speed using ONNX Runtime.

## Datasets
- **LTCXR**: LongtailCXR Test - 20k test samples,
- **RSNA-M**: RSNA Mammography STD - 10k test samples
- **MGBCXR**: 684 test samples, 80 classes used in test

## Models
- **Baseline (fp16)**: the original pytorch model.
- **OnnxRuntime (fp16)**: ONNX conversion of original model, with optimizations specific for OnnxRuntime.

## Metrics
- **Latency**: The time taken for the model to process a single input sample in milliseconds.
- **Speed-up**: The percentage reduction in latency compared to the baseline model.
- **mAUC**: Mean Area Under the Curve, a measure of the model's ability to distinguish between classes.

## Results
The table below shows the inference latency, speed-up, and accuracy of the model on various datasets. The speed-up is calculated as the percentage reduction in latency compared to the baseline model.

|**Dataset**        |**Model**        |**Latency** (ms) |**Speed-up** | **mAUC**    |
|-------------------|-----------------|-----------------|-------------|-------------|
| **LTCXR**         | **Baseline**    | 137             |  -          |   0.877     |
|                   | OnnxRuntime     | 45              | **61.84%**  |   0.877     |
| **RSNA-M**        | **Baseline**    | 85              |  -          |   0.888     |
|                   | OnnxRuntime     | 34              | **60.00%**  |   0.889     |
| **MGBCXR**        | **Baseline**    | 322             |  -          |   0.926     |
|                   | OnnxRuntime     | 84              | **73.91%**  |   0.928     |