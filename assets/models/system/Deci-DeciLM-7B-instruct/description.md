DeciLM-7B-instruct is a model for short-form instruction following, built by LoRA fine-tuning on the SlimOrca dataset. It is a derivative of the recently released DeciLM-7B language model, a pre-trained, high-efficiency generative text model with 7 billion parameters. DeciLM-7B-instruct is one of the best 7B instruct models obtained using simple LoRA fine-tuning, without relying on preference optimization techniques such as RLHF and DPO. DeciLM-7B-instruct is intended for commercial and research use in English. However, like all large language models, its outputs are unpredictable and may generate responses that are inaccurate, biased, or otherwise objectionable. Developers planning to use DeciLM-7B-instruct should undertake thorough safety testing and tuning designed explicitly for their intended applications of the model before deployment.



# Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


# Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>


# Sample inputs and outputs (for real-time inference)

### Sample input
```json
{
  "input_data": {
      "input_string": ["How do I make the most delicious pancakes the world has ever tasted?"],
      "parameters":{   
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
    "0": "How do I make the most delicious pancakes the world has ever tasted?\n\nAnswer: In a large bowl, whisk together the flour, sugar, baking powder, and salt. In a separate bowl, whisk together the milk, eggs, and melted butter. Pour the wet ingredients into the dry ingredients and stir until just combined. Add more milk if the batter seems too thick. Heat a non-stick pan or griddle over medium heat. Lightly grease the pan with butter or cooking spray. Pour about 1/4 cup of batter onto the"
  }
]
```