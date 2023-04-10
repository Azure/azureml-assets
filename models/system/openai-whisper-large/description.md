Whisper is an OpenAI pre-trained speech recognition model with potential applications for ASR solutions for developers. However, due to weak supervision and large-scale noisy data, it should be used with caution in high-risk domains. The model has been trained on 680k hours of audio data representing 98 different languages, leading to improved robustness and accuracy compared to existing ASR systems. However, there are disparities in performance across languages and the model is prone to generating repetitive texts, which may increase in low-resource languages. There are dual-use concerns and real economic implications with such performance disparities, and the model may also have the capacity to recognize specific individuals. The affordable cost of automatic transcription and translation of large volumes of audio communication is a potential benefit, but the cost of transcription may limit the expansion of surveillance projects.

> The above summary was generated using ChatGPT. Review the [original model card](https://huggingface.co/openai/whisper-large) to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|[asr-online-endpoint.ipynb](https://aka.ms/azureml-infer-online-sdk-asr)|[asr-online-endpoint.sh](https://aka.ms/azureml-infer-online-cli-asr)
Batch | coming soon


### Sample inputs and outputs (for real-time inference)

#### Sample input
```json
{
    "inputs": {
        "audio": ["https://audiovisionfiles.blob.core.windows.net/audio/audio.m4a", "https://audiovisionfiles.blob.core.windows.net/audio/audio.m4a"],
        "language": ["en", "fr"]
    }
}
```

#### Sample output
```json
[
    {
        "text": "This is a bright day."
    },
    {
        "text": "This is a bright day."
    }
]
```
