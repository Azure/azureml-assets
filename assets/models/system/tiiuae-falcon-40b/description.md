Falcon-40B is a large language model (LLM) developed by the Technology Innovation Institute (TII) with 40 billion parameters. It is a causal decoder-only model trained on 1 trillion tokens from the RefinedWeb dataset, enhanced with curated corpora. Falcon-40B supports English, German, Spanish, and French languages, with limited capabilities in Italian, Portuguese, Polish, Dutch, Romanian, Czech, and Swedish. It is available under the Apache 2.0 license.

Falcon-40B is considered the best open-source model currently available, optimized for inference with features such as FlashAttention and multiquery. However, it is recommended to fine-tune the model for specific use cases.

The training of Falcon-40B involved using 384 A100 40GB GPUs and took two months. The model carries biases and stereotypes encountered online and requires appropriate precautions for production use. It is suggested to finetune the model for specific tasks and consider guardrails. The technical specifications, training details, and evaluation results are provided in the summary.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/tiiuae/falcon-40b" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


### Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>


### Sample inputs and outputs (for real-time inference)

```json
{
    "input_data": {
        "input_string": ["Explain to me the difference between nuclear fission and fusion."]
    }
}
```

#### Sample output
```json
[
  {
    "0": "Nuclear fission and fusion are the 2 main ways that the nucleus of an atom can change. Nuclear fission occurs when a nucleus splits into 2 or more pieces. Nuclear fusion occurs when 2 or more nuclei join together to make a much larger nucleus.\n\nBoth nuclear fission and nuclear fusion take a lot of energy. For fission, you need to melt the nucleus to allow for the pieces to separate. For fusion, you need to sustain temperatures over 10 million degrees Celsius in order to combine 2 nuclei together.\n\nNuclear fission is how nuclear power plants create electricity. Nuclear fission involves using high-energy radiation to split a nucleus in 2 pieces. The pieces become unstable and quickly decay to release energy.\n\nNuclear fusion involves combining 2 or more nuclei of different elements into one new one. For this to happen, the nuclei need to be brought close together but kept apart by the enormous energies needed for nuclear fusion. In this process, a small amount of energy is released that can be converted to electricity. Nuclear fusion is how stars create the light and heat we feel from the sun.\n\nHuman beings can control which process happens in a nuclear reactor. We use catalysts to cause nuclear fission or nuclear fusion to happen.\n\nFusion happens naturally in the"
  }
]
```
