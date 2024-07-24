# **Model Details**

Model Name: Fuyu-8B
Developed by: Adept AI
Type: Multi-modal text and image transformer
License: CC-BY-NC
Availability: Available on Hugging Face
Key Features:
Simpler Architecture and Training:
Easier to understand, scale, and deploy.
Designed with a straightforward architecture and training procedure.
Digital Agent Support:
Tailored for digital agents.
Supports arbitrary image resolutions.
Capable of answering questions about graphs and diagrams.
Handles UI-based questions and fine-grained localization on screen images.
Speed:Fast response time - provides responses for large images in less than 100 milliseconds.
Performance:Despite optimization for a specific use-case, performs well on standard image understanding benchmarks.
Competent in visual question-answering and natural image captioning.

Training Approach:
Simplified approach treating image tokens like text tokens.
No image-specific position embeddings.
Raster-scan order used for feeding in as many image tokens as necessary.
Utilizes a special image-newline character to signify line breaks.
Adaptable to different image sizes using existing position embeddings.
Usage Recommendations:

Base model released; expects fine-tuning for specific use cases like verbose captioning or multimodal chat.
Responsive to few-shotting and fine-tuning for various applications.
Model Description:

Developed by: Adept AI
Model Type: Decoder-only multi-modal transformer model
License: CC-BY-NC
Description: Multi-modal model capable of processing both images and text inputs to produce textual outputs.
Note:
This summary provides an overview of Fuyu-8B, emphasizing its simplicity, digital agent support, speed, and versatile architecture. For detailed usage instructions, fine-tuning guidance, and specific model capabilities, it is recommended to refer to the model documentation on Hugging Face and Adept AI's resources.
Resources for more information: Check out our blog post-https://www.adept.ai/blog/fuyu-8b

# **Inference samples**

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


## **Sample inputs and outputs (for real-time inference)**

### **Sample input**
```json
{
    "input_data":{
       "input_string":["the meaning of life is"],
       "parameters":{
             "temperature":0.5,
             "top_p":0.5,
             "max_new_tokens":10,
              "do_sample":true
       }
    }
}
```

### **Sample output**
```json
[
  {
    "0": "the meaning of life is "a strong sense of purpose, " "
  }
]
```
