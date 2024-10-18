## Overview

The CXRReportGen model utilizes a multimodal architecture, integrating a BiomedCLIP image encoder with a Phi-3-Mini text encoder to help an application interpret complex medical imaging studies of chest X-rays. CXRReportGen follows the same framework as **[MAIRA-2](https://www.microsoft.com/en-us/research/publication/maira-2-grounded-radiology-report-generation/)**. When built upon and integrated into an application, CXRReportGen may help developers generate comprehensive and structured radiology reports, with visual grounding represented by bounding boxes on the images.

This repository contains the CXRReportGen model, which is packaged in MLflow format and deployed using Azure ML service. The estimated time to package and begin to build upon the model is approximately 1 hour. 

This model is intended and provided as-is for research and model development exploration. CXRReportGen is not designed or intended to be deployed in clinical settings as-is nor is it intended for use in the diagnosis or treatment of any health or medical condition (including generating radiology reports for use in patient care), and the model’s performance for such purposes has not been established. 
You bear sole responsibility and liability for any use of CXRReportGen, including verification of outputs and incorporation into any product or service intended for a medical purpose or to inform clinical decision-making, compliance with applicable healthcare laws and regulations, and obtaining any necessary clearances or approvals. 

For documentation and example Jupyter Notebooks, visit: https://aka.ms/CXRReportGenDocs.

### Training information

| **Training Dataset**   | **Details**        | 
|----------------|---------------------|
| **[MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/)**  | Frontal chest X-rays from the training partition of the MIMIC-CXR dataset and the associated text reports. Rule-based processing was carried out to extract findings and impressions separately, or to map non-labeled report sections to the relevant sections. During training, text is randomly sampled from either the findings or the impression section. In total 203,170 images from this dataset were used.|
| **Proprietary datasets**  | Multiple other proprietary datasets, composed of procured data, were additionally leveraged for training. Caution was taken to ensure there was no leakage of test data samples in the data used for training. |  

**Training Statistics:**
  - **Data Size:** ~400,000 samples
  - **Batch Size:** 16
  - **Epochs:** 3
  - **Learning Rate:** 2.5e-05
  - **Hardware:** 8 A100 GPUs
  - **Training Time:** 1 day and 19 hours
  - **Sku:** Standard_ND96amsr_A100_v4

### License and where to send questions or comments about the model
The license for CXRReportGen is the MIT license.
For questions or comments, please contact: hlsfrontierteam@microsoft.com

## Benchmark Results

### Findings Generation on MIMIC-CXR test set:

| CheXpert F1-14 (Micro) | CheXpert F1-5 (Micro)| RadGraph-F1 | ROUGE-L | BLEU-4|
|----------------|--------------|-------------|---------|-------|
| 59.1 | 59.7 | 40.8 | 39.1 |23.7 |


### Grounded Reporting on [GR-Bench test set](https://arxiv.org/pdf/2406.04449v1):

| CheXpert F1-14 (Micro) | RadGraph-F1 | ROUGE-L | Box-Completion (Precision/Recall)|
|------------------------|------------ |----------|-----------------|
| 60.0 | 55.6 | 56.6 | 71.5/82.0 |

## Carbon Footprint
The estimated carbon emissions during training are 0.06364 tCO2eq.


## Sample Input and Output

### Input:
```json
{'input_data': 
  {'columns': ['frontal_image', 'lateral_image', 'indication', 'technique', 'comparison'],
  'index': [0],
  'data': [
    [
      base64.encodebytes(read_image(frontal)).decode("utf-8"), 
      base64.encodebytes(read_image(lateral)).decode("utf-8"), 
      'Pneumonia', 
      'One view chest', 
      'None'
    ]]},
 'params': {}}
```

### Output:
Output is json encoded inside an array.
```python
findings = json.loads(result[0]["output"])
findings
```

```json
[['Cardiac silhouette remains normal in size.', None],
 ['Hilar contours are unremarkable.', None],
 ['There are some reticular appearing opacities in the left base not seen on the prior exam.',
  [[0.505, 0.415, 0.885, 0.775]]],
 ['There is blunting of the right costophrenic sulcus.',
  [[0.005, 0.555, 0.155, 0.825]]],
 ['Upper lungs are clear.', None]]
```
The generated bounding box coordinates are the (x, y) coordinates of the top left and bottom right corners of the box, e.g. (x_topleft, y_topleft, x_bottomright, y_bottomright). These are relative to the cropped image (that is, the image that the model ultimately got as input), so be careful while visualising.

You can optionally apply the below code on the output to adjust the size:
```python
  def adjust_box_for_original_image_size(box: BoxType, width: int, height: int) -> BoxType:
      """
      This function adjusts the bounding boxes from the MAIRA-2 model output to account for the image processor
      cropping the image to be square prior to the model forward pass. The box coordinates are adjusted to be
      relative to the original shape of the image assuming the image processor cropped the image based on the length
      of the shortest side.

      Args:
          box (BoxType):
              The box to be adjusted, normalised to (0, 1).
          width (int):
              Original width of the image, in pixels.
          height (int):
              Original height of the image, in pixels.

      Returns:
          BoxType: The box normalised relative to the original size of the image.
      """
      crop_width = crop_height = min(width, height)
      x_offset = (width - crop_width) // 2
      y_offset = (height - crop_height) // 2

      norm_x_min, norm_y_min, norm_x_max, norm_y_max = box

      abs_x_min = int(norm_x_min * crop_width + x_offset)
      abs_x_max = int(norm_x_max * crop_width + x_offset)
      abs_y_min = int(norm_y_min * crop_height + y_offset)
      abs_y_max = int(norm_y_max * crop_height + y_offset)

      adjusted_norm_x_min = abs_x_min / width
      adjusted_norm_x_max = abs_x_max / width
      adjusted_norm_y_min = abs_y_min / height
      adjusted_norm_y_max = abs_y_max / height

      return (adjusted_norm_x_min, adjusted_norm_y_min, adjusted_norm_x_max, adjusted_norm_y_max)
```

## Ethical Considerations

CXRReportGen is not designed or intended to be deployed as-is in clinical settings: for use in the diagnosis, cure, mitigation, treatment, or prevention of disease or other conditions; for use as a substitute of professional medical advice, diagnosis, treatment, or clinical judgment of a healthcare professional; or to generate draft radiology reports for use in patient care.  

Microsoft believes Responsible AI is a shared responsibility and we have identified six principles and practices help organizations address risks, innovate, and create value: fairness, reliability and safety, privacy and security, inclusiveness, transparency, and accountability. When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure this model meets requirements for the relevant use case and addresses unforeseen product misuse.   

While testing the model with images and/or text, ensure the the data is PHI free and that there are no patient information or information that can be tracked to a patient identity.

For detailed guidelines on ethical use, refer to Microsoft’s [Responsible AI Principles](https://www.microsoft.com/en-us/ai/responsible-ai)

## Hardware Requirement for Compute Instances
- Supports CPU and GPU
- Default: Single A100 GPU or Intel CPU
- Minimum: Single GPU instance with 24Gb Memory (Fastest) or CPU
