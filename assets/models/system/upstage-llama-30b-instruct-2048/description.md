# **Model Details**

Name: upstage-llama-30b-instruct-2048

Model Details:
Developed by: Upstage
Backbone Model: LLaMA
Variations: 30B/1024, 30B/2048, 65B/1024 with different model parameter sizes and sequence lengths
Language(s): English
Library: HuggingFace Transformers
License: Non-commercial Bespoke License, governed by the Meta license. Access granted through a form submission for those who have lost weights or encountered conversion issues to Transformers format.

Usage:
Tested on: A100 80GB
Capability: Can handle up to 10k+ input tokens, facilitated by the rope_scaling option.
Hardware: Utilized an A100x8 * 1 for training.
Training Factors: Fine-tuned using a combination of the DeepSpeed library and the HuggingFace Trainer / HuggingFace Accelerate.

Evaluation Results:
Benchmark Datasets: Evaluated on ARC-Challenge, HellaSwag, MMLU, and TruthfulQA using the lm-evaluation-harness repository.

Ethical Considerations:
Ethical Issues: None reported. The model's training process did not involve the benchmark test set or the training set, addressing ethical concerns.

Contact Information:
Where to Send Comments: Feedback can be provided by opening an issue in the Hugging Face community's model repository.

Why Upstage LLM: The Upstage LLM research, particularly the 70B model, holds a leading position in openLLM rankings as of August 1st. The model invites businesses to implement private LLM, offering fine-tuning options with personalized data. Contact details are provided for a tailored solution.
In summary, Upstage's LLaMa-30b-instruct-2048 is an English language model with various parameter sizes and sequence lengths. Access is granted through a form submission due to licensing restrictions. The model has been tested on specific hardware, fine-tuned using a combination of libraries, and evaluated on benchmark datasets. Ethical considerations are addressed, and the model's success is highlighted, inviting businesses to explore private LLM implementation.

For More details reach out to us: https://www.upstage.ai/


# **Inference samples**

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon

# Sample inputs and outputs (for real-time inference)

#### Sample input
```json
[
  {
    "endpoint_name": "upstage-llama-3-Standard-ND96ams",
    "deployment_name": "default",
    "inference_payload": {
      "input_data": {
        "input_string": [
          "My name is John and I am",
          "Once upon a time,"
        ],
        "parameters": {
          "max_new_tokens": 25,
          "do_sample": true,
          "temperature": 0.5,
          "top_p": 0.5
        }
      }
    },
    "response": "[{\"0\": \"My name is John and I am a professional musician. I have been playing the guitar for over 20 years and teaching for 10 years.\"}, {\"0\": \"Once upon a time, there was a little girl named Emily. Emily loved to play with her toys, especially her teddy bear,\"}]",
    "sku": "Standard_ND96amsr_A100_v4",
    "status": "Completed"
  }
]
```