## Model Description
# Mixtral-8x22B


Converted to HuggingFace Transformers format using the script [here](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1/blob/main/convert.py).

The Mixtral-8x22B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts.

Mixtral-8x22B-v0.1 is a pretrained base model and therefore does not have any moderation mechanisms.
# The Mistral AI Team
Albert Jiang, Alexandre Sablayrolles, Alexis Tacnet, Antoine Roux, Arthur Mensch, Audrey Herblin-Stoop, Baptiste Bout, Baudouin de Monicault,Blanche Savary, Bam4d, Caroline Feldman, Devendra Singh Chaplot, Diego de las Casas, Eleonore Arcelin, Emma Bou Hanna, Etienne Metzger, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Harizo Rajaona, Jean-Malo Delignon, Jia Li, Justus Murke, Louis Martin, Louis Ternon, Lucile Saulnier, Lélio Renard Lavaud, Margaret Jennings, Marie Pellat, Marie Torelli, Marie-Anne Lachaux, Nicolas Schuhl, Patrick von Platen, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Thibaut Lavril, Timothée Lacroix, Théophile Gervet, Thomas Wang, Valera Nemychnikova, William El Sayed, William Marshall.
# [Open LLM Leaderboard Evaluation Results](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
Detailed results can be found [here](https://huggingface.co/datasets/open-llm-leaderboard/details_mistral-community__Mixtral-8x22B-v0.1)

|             Metric              |Value|
|---------------------------------|----:|
|Avg.                             |74.46|
|AI2 Reasoning Challenge (25-Shot)|70.48|
|HellaSwag (10-Shot)              |88.73|
|MMLU (5-Shot)                    |77.81|
|TruthfulQA (0-shot)              |51.08|
|Winogrande (5-shot)              |84.53|
|GSM8k (5-shot)                   |74.15|



# **Inference samples**

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon

### Sample inputs and outputs

#### Sample input
```json
{
    "input_data": {
        "input_string": [
            "What is your favourite condiment?",
            "Do you have mayonnaise recipes?"
        ],
        "parameters": {
            "max_new_tokens": 100,
            "do_sample": true,
            "return_full_text": false
        }
    }
}
```

#### Sample output
```json
[
  {
    "0": "\n\nDoes Hellmann's Mayonnaise Mallows really exist?\n\nThis is a difficult one because I want to pick Orkney ice cream which is unbelievable but I am also drawn to Hellmann's Mayonnaise Mallows (yeah, they really do exist) which I recently tried for the first time in California.\n\nThey were exactly how I expected them to taste – like marshmallows made from mayonnaise. I can'",
    "1": " I would imagine that the ingredients consist, at least in large part, of oil and cream [suggest edit]. However, I'm interested in baking mayonnaise into food, which means I'm worried that 50% of mayonnaise is just going to turn into oil and get absorbed by whatever it's cooked with [suggest edit] [suggest edit]. I thought that perhaps there might be a different recipe for mayonnaise which could be used specifically to with"
  }
]
```
