The RoBERTa Large model is a large transformer-based language model that was developed by the Hugging Face team. It is pre-trained on masked language modeling and can be used for tasks such as sequence classification, token classification, or question answering. Its primary usage is as a fine-tuning tool and is case-sensitive. Additionally, there are metrics provided for DistilBART models, including the number of parameters, inference time, speedup, Rouge 2, and Rouge-L. The distilbart-xsum-12-6 model is recommended with 306 million parameters, 137 milliseconds inference time, 1.68 speedup, 22.12 Rouge 2, and 36.99 Rouge-L.


> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/sshleifer/distilbart-cnn-12-6" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-summarization" target="_blank">summarization-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-summarization" target="_blank">summarization-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-summarization" target="_blank">summarization-batch-endpoint.ipynb</a>| coming soon


### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|--|--|--|--|--|
Summarization|News Summary|<a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank">CNN DailyMail</a>|<a href="https://aka.ms/azureml-ft-sdk-news-summary" target="_blank">news-summary.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-news-summary" target="_blank">news-summary.sh</a>
Translation|Translate English to Romanian|<a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank">WMT16</a>|<a href="https://aka.ms/azureml-ft-sdk-translation" target="_blank">translate-english-to-romanian.ipynb</a>|<a href="https://aka.ms/azureml-ft-cli-translation" target="_blank">translate-english-to-romanian.sh</a>


### Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Summarization | Summarization | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank">cnn_dailymail</a> | <a href="https://aka.ms/azureml-eval-sdk-summarization" target="_blank">evaluate-model-summarization.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-summarization" target="_blank">evaluate-model-summarization.yml</a>


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "input_data": {
        "input_string": ["The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.", "Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres (41 square miles). The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880, or about 18 percent of the population of France as of 2017."]
    }
}
```

#### Sample output
```json
[
  {
    "0": " The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building . It was the first structure to reach a height of 300 metres . It is now taller than the Chrysler Building in New York City by 5.2 metres (17 ft) Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France ."
  },
  {
    "0": " Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018 . The city is the centre and seat of government of the region and province of Île-de-France, or Paris Region . Paris Region has an estimated 18 percent of the population of France as of 2017 ."
  }
]
```
