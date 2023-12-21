Whisper is an OpenAI pre-trained speech recognition model with potential applications for ASR solutions for developers. However, due to weak supervision and large-scale noisy data, it should be used with caution in high-risk domains. The model has been trained on 680k hours of audio data representing 98 different languages, leading to improved robustness and accuracy compared to existing ASR systems. However, there are disparities in performance across languages and the model is prone to generating repetitive texts, which may increase in low-resource languages. There are dual-use concerns and real economic implications with such performance disparities, and the model may also have the capacity to recognize specific individuals. The affordable cost of automatic transcription and translation of large volumes of audio communication is a potential benefit, but the cost of transcription may limit the expansion of surveillance projects.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/openai/whisper-large" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-asr" target="_blank">asr-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-asr" target="_blank">asr-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-asr" target="_blank">asr-batch-endpoint.ipynb</a>| coming soon


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "input_data": {
        "audio": ["https://datasets-server.huggingface.co/assets/librispeech_asr/--/all/train.clean.100/84/audio/audio.mp3"],
        "language": ["en"]
    }
}
```

#### Sample output
```json
[
    {
        "text": "Since that day, he had never been heard of. In vain, Marguerite dismissed her guests, changed her way of life. The Duke was not to be heard of. I was the gainer in so."
    }
]
```
