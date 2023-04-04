GPT-2 is a transformer-based language model intended for AI researchers and practitioners. It was trained on unfiltered content from Reddit and may have biases. It is best used for text generation, but the training data has not been publicly released. It has several limitations and should be used with caution in situations that require truth and in systems that interact with humans. There are different versions of the model, including GPT-Large, GPT-Medium, and GPT-Xl, available for different use cases. The information provided by the OpenAI team is to complete and give specific examples of bias in their model.


> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/gpt2) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[text-generation-online-endpoint.ipynb](https://aka.ms/azureml-infer-online-sdk-text-generation)|[text-generation-online-endpoint.sh](https://aka.ms/azureml-infer-online-cli-text-generation)
Batch | todo


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Text Classification|Emotion Detection|[Emotion](https://huggingface.co/datasets/dair-ai/emotion)|[emotion-detection.ipynb](https://aka.ms/azureml-ft-sdk-emotion-detection)|[emotion-detection.sh](https://aka.ms/azureml-ft-cli-emotion-detection)
Token Classification|Token Classification|[Conll2003](https://huggingface.co/datasets/conll2003)|[token-classification.ipynb](https://aka.ms/azureml-ft-sdk-token-classification)|[token-classification.sh](https://aka.ms/azureml-ft-cli-token-classification)


### Model Evaluation

|Task|Use case|Dataset|Python sample (Notebook)|
|---|--|--|--|
|Text generation||todo|[evaluate-model-text-generation.ipynb](https://aka.ms/azureml-eval-sdk-text-generation/)|


### Sample inputs and outputs (for real-time inference)

#### Sample input
```
{
    "inputs": {
        "input_string": ["My name is John and I am", "Once upon a time,"]
    },
    "parameters": {
        "min_length": 20,
        "max_length": 30,
        "num_return_sequences": 2
    }
}
```

#### Sample output
```
[
    [
        {
            "generated_text": "My name is John and I am a student at UC Berkeley. It is my main interest to do research in the humanities. I am going to share"
        },
        {
            "generated_text": "My name is John and I am from West London. But the 31-year-old was left stunned after his video appeared on Reddit last"
        }
    ],
    [
        {
            "generated_text": "Once upon a time, they were just another small family, only three. She says one day that her father was getting a new license"
        },
        {
            "generated_text": "Once upon a time, my character had the power to grant a certain amount of protection and to change the form of the power to that of the caster"
        }
    ]
]
```
