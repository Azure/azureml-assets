# **Model Details**

The Austism/chronos-hermes-13b-v2 is a state-of-the-art language model that is a FP16 PyTorch / HF version of the chronos-13b-v2, which is based on the LLaMA v2 Base model. This model is designed to be used for further quantization or for running in full precision, provided that sufficient VRAM is available.

The model is primarily designed for applications such as chatting, role-playing, and storywriting, and it excels in maintaining good reasoning and logic. One of the key features of Chronos is its ability to generate very long outputs with coherent text. This is largely due to the human inputs it was trained on, which allows it to maintain context over extended passages of text.

Furthermore, it supports a context length of up to 4096 tokens, making it capable of understanding and generating responses for longer conversations or narratives. This makes it particularly useful for applications that require maintaining context over long pieces of text or for generating extended pieces of content.

In summary, the Austism/chronos-hermes-13b-v2 model is a powerful tool for a wide range of language processing tasks, offering both precision and versatility. Whether you’re looking to engage in detailed role-play scenarios, write complex narratives, or simply have a chat, this model has you covered.

# Model Evaluation Sample

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>


# **Inference samples**

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon
