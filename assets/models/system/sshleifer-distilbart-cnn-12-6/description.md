The RoBERTa Large model is a large transformer-based language model that was developed by the Hugging Face team. It is pre-trained on masked language modeling and can be used for tasks such as sequence classification, token classification, or question answering. Its primary usage is as a fine-tuning tool and is case-sensitive. Additionally, there are metrics provided for DistilBART models, including the number of parameters, inference time, speedup, Rouge 2, and Rouge-L. The distilbart-xsum-12-6 model is recommended with 306 million parameters, 137 milliseconds inference time, 1.68 speedup, 22.12 Rouge 2, and 36.99 Rouge-L.


### Evaluation Samples

Task| Use case| Dataset| Python sample | CLI with YAML
|--|--|--|--|--|
Summarization | Summarization | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank">cnn_dailymail</a> | <a href="https://aka.ms/azureml-eval-sdk-summarization" target="_blank">evaluate-model-summarization.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-summarization" target="_blank">evaluate-model-summarization.yml</a>

### Inference samples

Inference type|Python sample |CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-summarization" target="_blank">summarization-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-summarization" target="_blank">summarization-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-summarization" target="_blank">summarization-batch-endpoint.ipynb</a>| coming soon

### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{ 

  "input_data": ["The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."]
} 
```

#### Sample output
```json
[ "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building . It was the first structure to reach a height of 300 metres . It is now taller than the Chrysler Building in New York City by 5.2 metres (17 ft) Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France ." ] 
```
