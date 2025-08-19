The model is optimized for inference speed using ONNX Runtime.

## Datasets
- **LTCXR**: LongtailCXR Test - 20k test samples,
- **RSNA-M**: RSNA Mammography STD - 10k test samples
- **RSNA-B-Male**: RSNA Bone Age Male - 773 test samples, 256 total classes, 65 classes used in test
- **Gastrovision**: 3.2k test samples, 27 classes
- **MGBCXR**: 684 test samples, 80 classes used in test

## Models
- **Baseline (fp16)**: the original pytorch model.
- **OnnxRuntime (fp16)**: ONNX conversion of original model, with optimizations specific for OnnxRuntime.

## Metrics
- **Latency**: The time taken for the model to process a single input sample in milliseconds.
- **Speed-up**: The percentage reduction in latency compared to the baseline model.
- **Accuracy**: The percentage of correct predictions made by the model.
- **BACC**: Balanced Accuracy, which accounts for class imbalance in the dataset.
- **mAUC**: Mean Area Under the Curve, a measure of the model's ability to distinguish between classes.

## Results
The table below shows the inference latency, speed-up, and accuracy of the model on various datasets. The speed-up is calculated as the percentage reduction in latency compared to the baseline model.

|**Dataset**        |**Model**        |**Latency** (ms) |**Speed-up** |**Accuracy** |**BACC**    |**mAUC**    |
|-------------------|-----------------|-----------------|-------------|-------------|------------|------------|
| **LTCXR**         | **Baseline**    | 137             |  -          | 11.31%      | 39.38%     | 87.73%     |
|                   | OnnxRuntime     | 45              | **61.84%**  | 11.13%      | 39.34%     | 87.72%     |
| **RSNA-M**        | **Baseline**    | 85              |  -          | 61.84%      | 66.5%      | 88.83%     |
|                   | OnnxRuntime     | 34              | **60.00%**  | 62.26%      | 66.69%     | 88.85%     |
| **RSNA-B-Male**   | **Baseline**    | 270             |  -          | 26.78%      | 15.26%     | 93.47%     |
|                   | OnnxRuntime     | 73              | **72.96%**  | 26.65%      | 15.11%     | 93.47%     |
| **Gastrovision**  | **Baseline**    | 160             |  -          | 20.92%      | 11.17%     | 77.61%     |
|                   | OnnxRuntime     | 50              | **68.75%**  | 21.04%      | 11.26%     | 76.77%     |
| **MGBCXR**        | **Baseline**    | 322             |  -          | 46.93%      | 34.61%     | 92.55%     |
|                   | OnnxRuntime     | 84              | **73.91%**  | 46.93%      | 34.61%     | 92.75%     |