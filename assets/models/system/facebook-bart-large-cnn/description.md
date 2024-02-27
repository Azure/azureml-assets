BART is a transformer model that combines a bidirectional encoder similar to BERT with an autoregressive decoder akin to GPT. It is trained using two main techniques: (1) corrupting text with a chosen noising function, and (2) training a model to reconstruct the original text.

When fine-tuned for specific tasks such as text generation (e.g., summarization, translation), BART demonstrates exceptional effectiveness. However, it also performs well on comprehension tasks like text classification and question answering. This specific checkpoint has undergone fine-tuning on CNN Daily Mail, a vast dataset consisting of text-summary pairs.


> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/facebook/bart-large-cnn" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-summarization" target="_blank">summarization-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-summarization" target="_blank">summarization-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-summarization" target="_blank">summarization-batch-endpoint.ipynb</a>| coming soon


### Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Summarization | Summarization | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank">cnn_dailymail</a> | <a href="https://aka.ms/azureml-eval-sdk-summarization" target="_blank">evaluate-model-summarization.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-summarization" target="_blank">evaluate-model-summarization.yml</a>


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "input_data": {
        "input_string": ["The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."]
    }
}
```

#### Sample output
```json
[ "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world." ] 
```
