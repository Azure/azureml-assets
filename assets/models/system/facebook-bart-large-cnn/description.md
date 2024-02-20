The BART model is a transformer encoder-encoder model trained on English language data, and fine-tuned on CNN Daily Mail. It is used for text summarization and has been trained to reconstruct text that has been corrupted using an arbitrary noising function. The model is effective for text generation tasks such as summarization, and works well for comprehension tasks such as text classification and question answering. It can be used with the pipeline API in Python, as detailed in the code snippet provided. Lastly, it was introduced in the paper "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension" by Lewis et al.


> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/facebook/bart-large-cnn" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

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
        "input_string": ["The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."]
    }
}
```

#### Sample output
```json
[ "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world." ] 
```
