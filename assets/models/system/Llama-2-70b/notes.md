## Intended Use

**Intended Use Cases** Llama 2 is intended for commercial and research use in English. Tuned models are intended for assistant-like chat, whereas pretrained models can be adapted for a variety of natural language generation tasks.

**Out-of-scope Uses** Use in any manner that violates applicable laws or regulations (including trade compliance laws).  , Use in languages other than English. Use in any other way that is prohibited by the Acceptable Use Policy and Licensing Agreement for Llama 2.

## Hardware and Software

**Training Factors** We used custom training libraries, Meta's Research Super Cluster, and production clusters for pretraining. Fine-tuning, annotation, and evaluation were also performed on third-party cloud compute.

**Carbon Footprint** Pretraining utilized a cumulative 3.3M GPU hours of computation on hardware of type A100-80GB (TDP of 350-400W). Estimated total emissions were 539 tCO2eq, 100% of which were offset by Meta’s sustainability program.

||Time (GPU hours)|Power Consumption (W)|Carbon Emitted(tCO<sub>2</sub>eq)|
|---|---|---|---|
|Llama 2 7B|184320|400|31.22|
|Llama 2 13B|368640|400|62.44|
|Llama 2 70B|1720320|400|291.42|
|Total|3311616||539.00|

**CO<sub>2</sub> emissions during pretraining.** Time: total GPU time required for training each model. Power Consumption: peak power capacity per GPU device for the GPUs used adjusted for power usage efficiency. 100% of the emissions are directly offset by Meta's sustainability program, and because we are openly releasing these models, the pretraining costs do not need to be incurred by others.

## Training Data

**Overview** Llama 2 was pretrained on 2 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets, as well as over one million new human-annotated examples. Neither the pretraining nor the fine-tuning datasets include Meta user data.

**Data Freshness** The pretraining data has a cutoff of September 2022, but some tuning data is more recent, up to July 2023. The pretraining data has a cutoff of September 2022, but some tuning data is more recent, up to July 2023.

## Ethical Considerations and Limitations

Llama 2 is a new technology that carries risks with use. Testing conducted to date has not, and could not, cover all scenarios, including uses in languages other than English. For these reasons, as with all LLMs, Llama 2’s potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate, biased or other objectionable responses to user prompts. Therefore, before deploying any applications of Llama 2, developers should perform safety testing and tuning tailored to their specific applications of the model.

Please see the Responsible Use Guide available at [https://ai.meta.com/llama/responsible-use-guide/](https://ai.meta.com/llama/responsible-use-guide)

## Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon

## Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Text Generation|Summarization|<a href="https://huggingface.co/datasets/samsum" target="_blank">Samsum</a>|<a href="https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/finetune/Llama-notebooks/text-generation/summarization_with_text_gen.ipynb" target="_blank">summarization_with_text_gen.ipynb</a>| <a href="https://github.com/Azure/azureml-examples/blob/main/cli/foundation-models/system/finetune/text-generation/text-generation.sh">text-generation.sh</a>
Text Classification|Emotion Detection|<a href="https://huggingface.co/datasets/dair-ai/emotion" target="_blank">Emotion</a>|<a href="https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/system/finetune/Llama-notebooks/text-classification/emotion-detection-llama.ipynb" target="_blank">emotion-detection-llama.ipynb</a>| <a href="https://github.com/Azure/azureml-examples/blob/main/cli/foundation-models/system/finetune/text-classification/emotion-detection.sh">emotion-detection.sh</a>

## Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>

## Sample inputs and outputs (for real-time inference)

### Supported Parameters

- temperature: Controls randomness in the model. Lower values will make the model more deterministic and higher values will make the model more random.
- max_new_tokens: The maximum number of tokens to generate.
- top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering. Default value is null, which disables top-k-filtering.
- top_p: The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling, defaults to null.
- do_sample: Whether or not to use sampling; use greedy decoding otherwise.
- return_full_text: Whether or not to return the full text (prompt + response) or only the generated part (response). Default value is false.
- ignore_eos: Whether to ignore the EOS token and continue generating tokens after the EOS token is generated. Defaults to False.

> List may not be complete.

### Sample input

```json
{ 
  "input_data": {
      "input_string": ["I believe the meaning of life is"],
      "parameters": {
            "top_p": 0.8,
            "temperature": 0.8,
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
        "0": "I believe the meaning of life is to be happy and to make other people happy.\nI think you only live once, so you have to make the best of it.\nI believe that the world is a very beautiful place, and that we should all try to make it a better place.\nI believe that we should all try to be kind to one another, and to help each other when we can.\nI believe that we should all try to be"
    }
]
```
