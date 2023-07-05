Falcon-7B is a large language model with 7 billion parameters. It is a causal decoder-only model developed by TII and trained on 1,500 billion tokens of RefinedWeb dataset, which was enhanced with curated corpora. The model is available under the Apache 2.0 license. It outperforms comparable open-source models and features an architecture optimized for inference. However, it is a raw, pretrained model that should be further finetuned for most use cases.

The model is recommended for research on large language models and as a foundation for further specialization and finetuning for specific tasks. It should not be used in production without adequate assessment of risks and mitigation. The model carries biases commonly encountered online and is trained on English and French data only.

The training details of Falcon-7B include information about the training data, training procedure, and hyperparameters used. It was trained on 384 A100 40GB GPUs using a 2D parallelism strategy combined with ZeRO. The model description mentions the architectural adaptations from the GPT-3 model, such as rotary positional embeddings, multiquery attention, and FlashAttention.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/tiiuae/falcon-7b" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


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
