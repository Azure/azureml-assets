The "Detect Defects" is a model designed for meticulous examination of images. It operates by employing GPT-4 Turbo with Vision to compare a test image against a reference image. Each analysis focuses on identifying variances or anomalies, classifying them as defects. This methodical comparison ensures that any discrepancies are accurately detected. The outcomes of these comparisons are then used to establish a comprehensive understanding of the product's quality, making it an essential tool in maintaining rigorous quality control standards.


### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://github.com/microsoft/promptflow/blob/pm/3p-inside-materials/docs/media/deploy-to-aml-code/sdk/deploy.ipynb" target="_blank">deploy-promptflow-model-python-example</a>|<a href="https://github.com/microsoft/promptflow/blob/pm/3p-inside-materials/docs/go-to-production/deploy-to-aml-code.md" target="_blank">deploy-promptflow-model-cli-example</a>
Batch | N/A | N/A

### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "inputs": {
        "reference_image": reference_image_path,
        "test_image": test_image_path,
        "system_message": "You're a professional defect detector. Your job is to compare the test image with reference image, please answer the question with \"No defect detected\" or \"Defect detected\", also explain your decision as detail as possible."
    }
}
```

#### Sample output
```json
{
    "outputs": {
        "response": "Defect detected. The test image shows a can lid with several noticeable defects when compared to the reference image. There are apparent signs of chipping or peeling on the surface around the tab of the can lid. The coating or material of the lid seems to have come off in patches, revealing the underlying layer or base material. This type of damage could compromise the integrity of the seal or lead to contamination of the can's contents. The reference image shows a pristine can lid with a smooth, uniform finish and no signs of damage, which is in contrast to the damaged state of the test image."
    }
}
```