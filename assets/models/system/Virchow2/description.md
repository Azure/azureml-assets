Virchow2 is a self-supervised vision transformer pretrained using 3.1M whole slide histopathology images. The model can be used as a tile-level feature extractor (frozen or finetuned) to achieve state-of-the-art results for a wide variety of downstream computational pathology use cases.


## Model Details

**Developed by:** Paige, NYC, USA and Microsoft Research, Cambridge, MA USA
**Model Type:** Image feature backbone
**Model Stats:**
    Params (M): 632
    Image size: 224 x 224
    Model Architecture:
**Architecture:** ViT-H/14
    Patch size: 14
    Layers: 32
    Embedding dimension: 1280
    Activation function: SwiGLU
    Attention heads: 16
    LayerScale: true
    Register tokens: 4
**Training Details:**
    Precision: Mixed precision (fp16)
    Objective: Modified DINOv2 (https://doi.org/10.48550/arXiv.2304.07193)
        KoLeo regularizer replaced with kernel density estimator
        Crop-and-resize augmentation replaced with extended context translation
**Paper:**
    Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology https://arxiv.org/pdf/2408.00738
**Pretraining Dataset:** Internal dataset of 3.1 million whole slide images from Memorial Sloan Kettering Cancer Center, tiles sampled at 2.0, 1.0, 0.5 and 0.25 microns per pixel resolution (5x, 10x, 20x, and 40x magnification).
**License:** CC-BY-NC-ND-4.0

## Model Usage

**Direct use**
Virchow2 intended to be used as a frozen feature extractor as the foundation for tile-level and whole slide-level classifiers.

**Downstream use**
Virchow2 can be finetuned to adapt to specific tasks and/or datasets.

**Terms of use**

The Virchow2 Model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the Virchow2 Model and its derivatives, which include models trained on outputs from the Virchow2 Model or datasets created from the Virchow2 Model, is prohibited and requires prior approval. By downloading /deploying the Virchow2 Model, you attest that all account information (affiliation, research use) is correct and up-to-date. By downloading/deploying the Virchow2 Model, you agree not to distribute, publish or reproduce a copy of the Virchow2 Model. If another user within your organization wishes to use the Virchow2 Model, they must register as an individual user and agree to comply with these terms of use. If you are a commercial entity, please contact the corresponding author. Further, by downloading/deploying the Virchow2 Model, you agree you will only use the Virchow2 Model for academic research purposes and will not use, or allow others to use, the Virchow2 Model to:

1.	Diagnose, cure, mitigate, treat, or prevent disease or any other conditions, including for Investigational Use Only (“IUO”), Research Use Only (“RUO”), commercial, clinical or other similar use, and including as a substitute for professional medical advice, a healthcare opinion, a diagnosis, treatment, or the clinical judgment of a healthcare professional, as no license or right is granted for any such purposes.
2.	Re-identify the deidentified data used to develop the Virchow2 Model;
3.	Violate the law or others’ rights, including to:
    a.	Engage in, promote, generate, contribute to, encourage, plan, incite, or further illegal or unlawful activity or content;
    b.	Engage in, promote, incite, or facilitate the harassment, abuse, threatening, or bullying of individuals or groups of individuals;
    c.	Engage in, promote, incite, or facilitate discrimination or other unlawful or harmful conduct in the provision of employment, employment benefits, credit, housing, other economic benefits, or other              essential goods and services;
    d.	Engage in the unauthorized or unlicensed practice of any profession including, but not limited to, financial, legal, medical/health, or related professional practices;
    e.	Collect, process, disclose, generate, or infer the identity of individuals or the health, demographic, or other sensitive personal or private information about individuals without rights and consents            required by applicable laws;
    f.	Engage in or facilitate any action or generate any content that infringes, misappropriates, or otherwise violates any third-party rights, including the outputs or results of any products or services             using the Virchow2 Model or any related materials; and
    g.	Create, generate, or facilitate the creation of malicious code, malware, computer viruses or do anything else that could disable, overburden, interfere with or impair the proper working, integrity,              operation or appearance of a website or computer system.
4.	Engage in, promote, incite, facilitate, or assist in the planning or development of activities that present a risk of death or bodily harm to individuals, including the use of the Virchow2 Model as a medical device, clinical support, diagnostic tool, or other technology intended to be used in the diagnosis, cure, mitigation, treatment, or prevention of disease or other conditions, including for Investigational Use Only (“IUO”), Research Use Only (“RUO”), commercial, clinical or similar use; and
5.	Intentionally deceive or mislead others, including representing that the use of the Virchow2 Model or its outputs is human-generated.
Further, you agree that you will appropriately disclose to end users any known dangers of your AI system.


**Citation**

Please cite the following work if you used this model in your research.

Zimmermann, E., Vorontsov, E., Viret, J. et al. Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology. arXiv preprint arXiv:2408.00738 (2024).

```
@article{zimmermann2024virchow2,
  title={Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology}, 
  author={Eric Zimmermann and Eugene Vorontsov and Julian Viret and Adam Casson and Michal Zelechowski and George Shaikovski and Neil Tenenholtz and James Hall and Thomas Fuchs and Nicolo Fusi and Siqi Liu and Kristen Severson},
  journal={arXiv preprint arXiv:2408.00738},
  year={2024},
}
```
**Disclaimer**
Virchow2 has been developed for research purposes and is not intended for diagnosis of real patients or projection/prediction of future disease possibilities.

Fairness evaluation cannot be completed due to limitations in the metadata. Underlying biases of the training datasets may not be well characterized and may not be representative of all demographics.

## Sample Input and Output (for real-time inference)
### Sample Input

```json
{
  "input_data": {
    "columns": [
      "image"
    ],
    "index":[0],
    "data": [
      ["image1"]
   ]
 }
}
```
Note:
- "image1" and "image2" should be publicly accessible urls or strings in base64 format.

### Sample Output
```json
[
  {
    "output": [
      0.0, 0.0, 0.0, 0.0
    ]
  }
]
```
Output will be image embeddings.
