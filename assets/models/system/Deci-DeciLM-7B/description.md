DeciLM-7B is a decoder-only text generation model with 7.04 billion parameters, released by Deci under the Apache 2.0 license. It is the top-performing 7B base language model on the Open LLM Leaderboard and uses variable Grouped-Query Attention (GQA) to achieve a superior balance between accuracy and computational efficiency. The model's architecture was generated using Deci's proprietary Neural Architecture Search technology, AutoNAC. DeciLM-7B is intended for commercial and research use in English and can be fine-tuned for various tasks and languages. However, like all large language models, its outputs are unpredictable and may generate responses that are inaccurate, biased, or otherwise objectionable. Developers planning to use DeciLM-7B should undertake thorough safety testing and tuning designed explicitly for their intended applications of the model before deployment.



# Model Evaluation Sample

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>

# Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


# Sample inputs and outputs (for real-time inference)

### Sample input
```json
{
  "input_data": {
    "input_string": [
      "In a shocking finding, scientists discovered a herd of unicorns living in"
    ],
    "parameters": {
      "top_p": 0.95,
      "temperature": 0.6,
      "max_new_tokens": 100,
      "do_sample": true
    }
  }
}
```

### Sample output
```json
[
  {
    "0": "In a shocking finding, scientists discovered a herd of unicorns living in the Pacific Ocean. The discovery was made by a team of scientists who were studying the effects of climate change on the ocean. They found that the unicorns were using the ocean as a refuge from the warmer temperatures on land.\n\nA team of scientists from the University of California, San Diego, and the University of California, Los Angeles, made the discovery while studying the effects of climate change on the ocean. They found that the unicorns were using the ocean as a"
  }
]
```