# **Model Details**

Name: lmsys-vicuna-33b-v1.3

Model Details:
Developed by: LMSYS
Model Type: Auto-regressive language model based on the transformer architecture.
License: Non-commercial license
Finetuned from: LLaMA

Uses:
Primary use is for research on large language models and chatbots.
Intended users: Researchers and hobbyists in natural language processing, machine learning, and artificial intelligence.

Getting Started:
Command Line Interface: https://github.com/lm-sys/FastChat#vicuna-weights

Version: Vicuna v1.3
Training Method: Fine-tuned from LLaMA using supervised instruction fine-tuning.
Training Data: Approximately 125K conversations collected from ShareGPT.com.
More Details: Refer to the "Training Details of Vicuna Models" section in the paper's appendix.

Evaluation:
Benchmarking: Evaluated with standard benchmarks.
Human Preference: Human-preference evaluations conducted.
LLM-as-a-Judge: Evaluation using LLM as a judge.
Details: Refer to the paper and leaderboard for more information.

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
    "endpoint_name": "lmsys-vicuna-33-Standard-ND40rs",
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
    "response": "[{\"0\": \"My name is John and I am a 22 year old college student. I am currently a junior at the University of California, Los Angeles (UCLA\"}, {\"0\": \"Once upon a time, in a land far, far away, there was a little girl named Cinderella. Cinderella lived with her evil\"}]",
    "sku": "Standard_ND40rs_v2",
    "status": "Completed"
  }
]
```