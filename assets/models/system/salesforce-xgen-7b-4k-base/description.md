 The "xgen-7b-4k-base" model is a large-scale language model developed by Salesforce Research. It has been trained using the GPT-3.5 architecture, which is a state-of-the-art language model known for its ability to generate coherent and contextually relevant text. The "xgen-7b-4k-base" model has a base configuration of 7 billion parameters, which enables it to capture complex patterns and relationships in the text data it is trained on. This large parameter count contributes to its ability to generate high-quality and contextually accurate responses.

> The above summary was generated using ChatGPT. Review the <a href="https://huggingface.co/Salesforce/xgen-7b-4k-base" target="_blank">original model card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.

### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon


### Model Evaluation

Task| Use case| Dataset| Python sample (Notebook)| CLI with YAML
|--|--|--|--|--|
Text generation | Text generation | <a href="https://huggingface.co/datasets/cnn_dailymail" target="_blank"> cnn_dailymail </a> | <a href="https://aka.ms/azureml-eval-sdk-text-generation/" target="_blank">evaluate-model-text-generation.ipynb</a> | <a href="https://aka.ms/azureml-eval-cli-text-generation/" target="_blank">evaluate-model-text-generation.yml</a>


### Model Performance

SKU Name| Avg. Requests Per Second | Avg. Latency(s) 
|--|--|--|
Standard_E16s_v3 | 0.03 | 50.34
Standard_F32s_v2 | 0.03 | 59.1
Standard_NC24s_v3 | 1.08 | 34.9


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
