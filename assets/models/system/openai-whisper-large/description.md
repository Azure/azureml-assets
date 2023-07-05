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
    "inputs": {
        "audio": ["https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac", "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/2.flac"],
        "language": ["en", "en"]
    }
}
```

#### Sample output
```json
[
    {
        "text": "He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fattened sauce."
    },
    {
        "text": "Stuff it into you. His belly counseled him."
    }
]
```
