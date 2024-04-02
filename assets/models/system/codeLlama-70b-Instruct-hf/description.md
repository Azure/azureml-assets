# **Code Llama**
Code Llama is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. This is the repository for the 70B instruct-tuned version in the Hugging Face Transformers format. This model is designed for general code synthesis and understanding. Links to other models can be found in the index at the bottom.

|     | Base Model                                                                    | Python                                                                                      | Instruct                                                                                        |
| --- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 7B  | [codellama/CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf) | [codellama/CodeLlama-7b-Python-hf](https://huggingface.co/codellama/CodeLlama-7b-Python-hf) | [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) |
| 13B  | [codellama/CodeLlama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf) | [codellama/CodeLlama-13b-Python-hf](https://huggingface.co/codellama/CodeLlama-13b-Python-hf) | [codellama/CodeLlama-13b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf) |
| 34B  | [codellama/CodeLlama-34b-hf](https://huggingface.co/codellama/CodeLlama-34b-hf) | [codellama/CodeLlama-34b-Python-hf](https://huggingface.co/codellama/CodeLlama-34b-Python-hf) | [codellama/CodeLlama-34b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf) |
| 70B | [codellama/CodeLlama-70b-hf](https://huggingface.co/codellama/CodeLlama-70b-hf) | [codellama/CodeLlama-70b-Python-hf](https://huggingface.co/codellama/CodeLlama-70b-Python-hf) | [codellama/CodeLlama-70b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf)

## Model capabilities:

- [x] Code completion.
- [ ] Infilling.
- [x] Instructions / chat.
- [] Python specialist.

# Model Use
- To use this model, please make sure to install transformers.
- pip install transformers accelerate

**Variations** The 70B Instruct model uses a [different prompt template](https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf#chat_prompt) than the smaller versions. To use it with transformers, we recommend you use the built-in chat template:
You can also use the model for text or code completion. This examples uses transformers' pipeline interface:

**Chat prompt** CodeLlama 70B Instruct uses a different format for the chat prompt than previous Llama 2 or CodeLlama models. As mentioned above, the easiest way to use it is with the help of the tokenizer's chat template. If you need to build the string or tokens, manually.

## Model Details
*Note: Use of this model is governed by the Meta license. Meta developed and publicly released the Code Llama family of large language models (LLMs).

**Model Developers** Meta

**Variations** Code Llama comes in four model sizes, and three variants:

* Code Llama: base models designed for general code synthesis and understanding
* Code Llama - Python: designed specifically for Python
* Code Llama - Instruct: for instruction following and safer deployment

All variants are available in sizes of 7B, 13B, 34B, and 70B parameters.

**This repository contains the base version of the 70B parameters model.**

**Input** Models input text only.

**Output** Models generate text only.

**Model Architecture** Code Llama is an auto-regressive language model that uses an optimized transformer architecture. It was fine-tuned with up to 16k tokens. This variant does not support long context of up to 100k tokens.

**Model Dates** ode Llama and its variants have been trained between January 2023 and January 2024.

**Status** This is a static model trained on an offline dataset. Future versions of Code Llama - Instruct will be released as we improve model safety with community feedback.

**License** A custom commercial license is available at: [https://ai.meta.com/resources/models-and-libraries/llama-downloads/](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)

**Research Paper** More information can be found in the paper "[Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)" or its [arXiv page](https://arxiv.org/abs/2308.12950).

## Intended Use
**Intended Use Cases** Code Llama and its variants are intended for commercial and research use in English and relevant programming languages. The base model Code Llama can be adapted for a variety of code synthesis and understanding tasks, Code Llama - Python is designed specifically to handle the Python programming language, and Code Llama - Instruct is intended to be safer to use for code assistant and generation applications.

**Out-of-Scope Uses** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English. Use in any other way that is prohibited by the Acceptable Use Policy and Licensing Agreement for Code Llama and its variants.

## Hardware and Software
**Training Factors** We used custom training libraries. The training and fine-tuning of the released models have been performed Meta’s Research Super Cluster.

**Carbon Footprint** In aggregate, training all 12 Code Llama models required 1400K GPU hours of computation on hardware of type A100-80GB (TDP of 350-400W). Estimated total emissions were 228.55 tCO2eq, 100% of which were offset by Meta’s sustainability program.

## Evaluation Results

See evaluations for the main models and detailed ablations in Section 3 and safety evaluations in Section 4 of the research paper.

## Ethical Considerations and Limitations

Code Llama and its variants are a new technology that carries risks with use. Testing conducted to date has been in English, and has not covered, nor could it cover all scenarios. For these reasons, as with all LLMs, Code Llama’s potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate or objectionable responses to user prompts. Therefore, before deploying any applications of Code Llama, developers should perform safety testing and tuning tailored to their specific applications of the model.

Please see the Responsible Use Guide available available at [https://ai.meta.com/llama/responsible-user-guide](https://ai.meta.com/llama/responsible-user-guide).

# **Model Evaluation samples**

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>


# **Inference samples**
Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


# Sample inputs and outputs (for real-time inference)

### Sample input

```json
{
    "input_data": {
        "input_string": [
            "def fibonacci("
        ],
        "parameters": {
            "top_p": 0.9,
            "temperature": 0.2,
            "do_sample": true,
            "max_new_tokens": 200
        }
    }
}

```

### Sample output
```json
[
  {
    "0": "def fibonacci(n):\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\n\ndef main():\n    print(fibonacci(5))\n\n\nif __name__ == \"__main__\":\n    main()\n"
  }
]
```
