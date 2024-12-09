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
>
### Sample input

```json
{
  "input_data": {
    "input_string": [
      {
        "role": "user",
        "content": "I am going to Paris, what should I see?"
      },
      {
        "role": "assistant",
        "content": "Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world."
      },
      {
        "role": "user",
        "content": "What is so great about #1?"
      }
    ],
    "parameters": {
      "temperature": 0.6,
      "top_p": 0.9,
      "do_sample": true,
      "max_new_tokens": 200
    }
  }
}
```

### Sample output

```json
{
  "output": "There are many reasons why the Eiffel Tower is one of the most iconic landmarks in Paris and a must-see attraction for visitors. Here are a few of the reasons why it's so great:\n\n1. Unique Design: The Eiffel Tower is an engineering marvel, with its unique design and shape that sets it apart from other landmarks around the world.\n2. Scale: The Eiffel Tower is an enormous structure, standing at over 1,000 feet tall, and offers breathtaking views of the city from its top platform.\n3. History: The Eiffel Tower was built for the World's Fair in 1889 and has since become a symbol of Paris and French culture.\n4. Romantic Atmosphere: The Eiffel Tower is often called the most romantic spot in Paris, and is a popular spot for couples to visit and take in"
}
```
