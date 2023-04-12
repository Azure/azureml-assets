The BART model is a transformer encoder-encoder model trained on English language data, and fine-tuned on CNN Daily Mail. It is used for text summarization and has been trained to reconstruct text that has been corrupted using an arbitrary noising function. The model is effective for text generation tasks such as summarization, and works well for comprehension tasks such as text classification and question answering. It can be used with the pipeline API in Python, as detailed in the code snippet provided. Lastly, it was introduced in the paper "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension" by Lewis et al.


> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/facebook/bart-large-cnn) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[summarization-online-endpoint.ipynb](https://aka.ms/azureml-infer-online-sdk-summarization)|[summarization-online-endpoint.sh](https://aka.ms/azureml-infer-online-cli-summarization)
Batch | coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Summarization|Summarization|[CNN DailyMail](https://huggingface.co/datasets/cnn_dailymail)|[news-summary.ipynb](https://aka.ms/azureml-ft-sdk-news-summary)|[news-summary.sh](https://aka.ms/azureml-ft-cli-news-summary)
Translation|Translation|[WMT16](https://huggingface.co/datasets/cnn_dailymail)|[translation.ipynb](https://aka.ms/azureml-ft-sdk-translation)|[translation.sh](https://aka.ms/azureml-ft-cli-translation)


### Model Evaluation

| Task          | Use case      | Dataset                                                        | Python sample (Notebook)                                                            | CLI with YAML                                                                     |
|---------------|---------------|----------------------------------------------------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| Summarization | Summarization | [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) | [evaluate-model-summarization.ipynb](https://aka.ms/azureml-eval-sdk-summarization) | [evaluate-model-summarization.yml](https://aka.ms/azureml-eval-cli-summarization) |


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "inputs": {
        "input_string": ["The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.", "Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres (41 square miles). The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880, or about 18 percent of the population of France as of 2017."]
    },
    "parameters": {
        "min_length": 20,
        "max_length": 70
    }
}
```

#### Sample output
```json
[
    {
        "summary_text": "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Its base is square, measuring 125 metres (410 ft) on each side. It is the second tallest free-standing structure in France after the Millau Viaduct."
    },
    {
        "summary_text": "Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018. The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region."
    }
]
```
