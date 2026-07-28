We report the results under completion format for Phi-3-Mini-4K-Instruct on standard open-source benchmarks measuring the model's reasoning ability (both common sense reasoning and logical reasoning). We compare to Mistral-7b-v0.1, Mixtral-8x7b, Gemma 7B, Llama-3-8B-Instruct, and GPT3.5-Turbo-1106.

All the reported numbers are produced with the exact same pipeline to ensure that the numbers are comparable. These numbers might differ from other published numbers due to slightly different choices in the evaluation.

As is now standard, we use few-shot prompts to evaluate the models, at temperature 0. 
The prompts and number of shots are part of a Microsoft internal tool to evaluate language models, and in particular we did no optimization to the pipeline for Phi-3.
More specifically, we do not change prompts, pick different few-shot examples, change prompt format, or do any other form of optimization for the model.

The number of k–shot examples is listed per-benchmark. 

| Category | Benchmark | Phi-3-Mini-4K-Ins | Gemma-7B | Mistral-7b | Mixtral-8x7b | Llama-3-8B-Ins | GPT3.5-Turbo-1106 |
|:----------|:-----------|:-------------------|:----------|:------------|:--------------|:----------------|:-------------------|
| Popular aggregated benchmark | AGI Eval <br>5-shot| 39.0 | 42.1 | 35.1 | 45.2 | 42 | 48.4 |
| | MMLU <br>5-shot | 70.9 | 63.6 | 61.7 | 70.5 | 66.5 | 71.4 |
| | BigBench Hard CoT<br>3-shot| 73.5 | 59.6 | 57.3 | 69.7 | 51.5 | 68.3 |
| Language Understanding | ANLI <br>7-shot | 53.6 | 48.7 | 47.1 | 55.2 | 57.3 | 58.1 |
| | HellaSwag <br>5-shot| 75.3 | 49.8 | 58.5 | 70.4 | 71.1 | 78.8 |
| Reasoning | ARC Challenge <br>10-shot | 86.3 | 78.3 | 78.6 | 87.3 | 82.8 | 87.4 |
| | BoolQ <br>0-shot | 78.1 | 66 | 72.2 | 76.6 | 80.9 | 79.1 |
| | MedQA <br>2-shot| 56.5 | 49.6 | 50 | 62.2 | 60.5 | 63.4 |
| | OpenBookQA <br>10-shot| 82.2 | 78.6 | 79.8 | 85.8 | 82.6 | 86 |
| | PIQA <br>5-shot| 83.5 | 78.1 | 77.7 | 86 | 75.7 | 86.6 |
| | GPQA <br>0-shot| 30.6 | 2.9 | 15 | 6.9 | 32.4 | 30.8 |
| | Social IQA <br>5-shot| 77.6 | 65.5 | 74.6 | 75.9 | 73.9 | 68.3 |
| | TruthfulQA (MC2) <br>10-shot| 64.7 | 52.1 | 53 | 60.1 | 63.2 | 67.7 |
| | WinoGrande <br>5-shot| 71.6 | 55.6 | 54.2 | 62 | 65 | 68.8 |
| Factual Knowledge | TriviaQA <br>5-shot| 61.4 | 72.3 | 75.2 | 82.2 | 67.7 | 85.8 |
| Math | GSM8K CoT <br>8-shot| 85.7 | 59.8 | 46.4 | 64.7 | 77.4 | 78.1 |
| Code Generation | HumanEval <br>0-shot| 57.3 | 34.1 | 28.0 | 37.8 | 60.4 | 62.2 |
| | MBPP <br>3-shot| 69.8 | 51.5 | 50.8 | 60.2 | 67.7 | 77.8 |
| **Average** | | **67.6** | **56.0** | **56.4** | **64.4** | **65.5** | **70.4** |


We take a closer look at different categories across 100 public benchmark datasets at the table below: 

| Category | Phi-3-Mini-4K-Instruct | Gemma-7B | Mistral-7B | Mixtral 8x7B | Llama-3-8B-Instruct | GPT-3.5-Turbo |
|:----------|:------------------------|:----------|:------------|:--------------|:---------------------|:---------------|
| Popular aggregated benchmark | 61.1 | 59.4 | 56.5 | 66.2 | 59.9 | 67.0 |
| Reasoning | 70.8 | 60.3 | 62.8 | 68.1 | 69.6 | 71.8 |
| Language understanding | 60.5 | 57.6 | 52.5 | 66.1 | 63.2 | 67.7 |
| Code generation | 60.7 | 45.6 | 42.9 | 52.7 | 56.4 | 70.4 |
| Math | 50.6 | 35.8 | 25.4 | 40.3 | 41.1 | 52.8 |
| Factual knowledge | 38.4 | 46.7 | 49.8 | 58.6 | 43.1 | 63.4 |
| Multilingual | 56.7 | 66.5 | 57.4 | 66.7 | 66.6 | 71.0 |
| Robustness | 61.1 | 38.4 | 40.6 | 51.0 | 64.5 | 69.3 |


Overall, the model with only 3.8B-param achieves a similar level of language understanding and reasoning ability as much larger models. However, it is still fundamentally limited by its size for certain tasks. The model simply does not have the capacity to store too much world knowledge, which can be seen for example with low performance on TriviaQA. However, we believe such weakness can be resolved by augmenting Phi-3-Mini with a search engine.   