Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. They are text-to-text, decoder-only large language models, available in English, with open weights, pre-trained variants, and instruction-tuned variants. Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning. Their relatively small size makes it possible to deploy them in environments with limited resources such as a laptop, desktop or your own cloud infrastructure, democratizing access to state of the art AI models and helping foster innovation for everyone.

# Training Details

## Training Dataset

These models were trained on a dataset of text data that includes a wide variety of sources, totaling 6 trillion tokens. Here are the key components:

<li>Web Documents: A diverse collection of web text ensures the model is exposed to a broad range of linguistic styles, topics, and vocabulary. Primarily English-language content.
<li>Code: Exposing the model to code helps it to learn the syntax and patterns of programming languages, which improves its ability to generate code or understand code-related questions.
<li>Mathematics: Training on mathematical text helps the model learn logical reasoning, symbolic representation, and to address mathematical queries.
The combination of these diverse data sources is crucial for training a powerful language model that can handle a wide variety of different tasks and text formats.

## Data Preprocessing
Here are the key data cleaning and filtering methods applied to the training data:

<li>CSAM Filtering: Rigorous CSAM (Child Sexual Abuse Material) filtering was applied at multiple stages in the data preparation process to ensure the exclusion of harmful and illegal content
<li>Sensitive Data Filtering: As part of making Gemma pre-trained models safe and reliable, automated techniques were used to filter out certain personal information and other sensitive data from training sets.
<li>Additional methods: Filtering based on content quality and safely in line with our policies.

## Hardware and Software
The Gemma model was implemented using Tensor Processing Unit (TPU) hardware, specifically TPUv5e. TPUs offer advantages such as enhanced performance, large memory capacity, scalability through TPU Pods, and cost-effectiveness for training large language models (LLMs). The use of TPUs aligns with Google's sustainability goals.

The software employed for training Gemma includes JAX and ML Pathways. JAX enables efficient utilization of hardware, including TPUs, for faster training of large models. ML Pathways, a Google initiative, focuses on creating artificially intelligent systems capable of generalizing across multiple tasks. The combination of JAX and ML Pathways simplifies the development workflow, as described in the Gemini family of models paper. The 'single controller' programming model of JAX and Pathways allows a single Python process to orchestrate the entire training run.

# Evaluation Results
The results of ethics and safety evaluations are within acceptable thresholds for meeting internal policies for categories such as child safety, content safety, representational harms, memorization, large-scale harms. On top of robust internal evaluations, the results of well known safety benchmarks like BBQ, BOLD, Winogender, Winobias, RealToxicity, and TruthfulQA are shown here.

| Benchmark	|Metric	| 2B Params	|7B Params|
|:----:|:----:|:----:|:----:|
|RealToxicity	|average	|6.86	|7.90|
|BOLD	|	|45.57	|49.08|
|CrowS-Pairs	|top-1	|45.82	|51.33|
|BBQ Ambig	|1-shot, top-1|	62.58	|92.54|
|BBQ Disambig	|top-1	|54.62	|71.99|
|Winogender	|top-1	|51.25	|54.17|
|TruthfulQA	| |	44.84|	31.81|
|Winobias 1_2	| |	56.12	|59.09|
|Winobias 2_2|		|91.10 | 92.23|
|Toxigen	| |	29.77|	39.59|




# License

<a href="https://ai.google.dev/gemma/terms" target="_blank">gemma-terms-of-use</a>


# Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-infer-online-sdk-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.ipynb</a>|<a href="https://aka.ms/azureml-infer-online-cli-text-generation-dolly" target="_blank">text-generation-online-endpoint-dolly.sh</a>
Batch |<a href="https://aka.ms/azureml-infer-batch-sdk-text-generation" target="_blank">text-generation-batch-endpoint.ipynb</a>| coming soon



# Sample input and output

### Sample input

```json
{
    "input_data": {
        "input_string": [
            "how does brain work?"
        ],
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.5,
            "max_new_tokens": 100,
            "do_sample": true
        }
    }
}
```



### Sample output

```json
[
  {
    "0": "how does brain work?\n\nThe brain is the most complex organ in the human body. It is responsible for controlling all of the body’s functions, including breathing, heart rate, and digestion. The brain is also responsible for thinking, feeling, and remembering.\n\nThe brain is made up of billions of cells, called neurons. Neurons are connected to each other by tiny fibers called axons. Axons carry electrical signals from one neuron to another.\n\nThe brain is divided into two main parts: the cerebrum and"
  }
]
```

