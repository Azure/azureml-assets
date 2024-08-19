We report the results for Phi-3-Medium-128k-Instruct on standard open-source benchmarks measuring the model's reasoning ability (both common sense reasoning and logical reasoning). We compare to Mixtral-8x22b, Gemini-Pro, Command R+ 104B, Llama-3-70B-Instruct, GPT-3.5-Turbo-1106, and GPT-4-Turbo-1106(Chat).

All the reported numbers are produced with the exact same pipeline to ensure that the numbers are comparable. These numbers might differ from other published numbers due to slightly different choices in the evaluation.

As is now standard, we use few-shot prompts to evaluate the models, at temperature 0. 
The prompts and number of shots are part of a Microsoft internal tool to evaluate language models, and in particular we did no optimization to the pipeline for Phi-3.
More specifically, we do not change prompts, pick different few-shot examples, change prompt format, or do any other form of optimization for the model.

The number of k–shot examples is listed per-benchmark. 

|Benchmark|Phi-3-Medium-128k-Instruct<br>14b|Command R+<br>104B|Mixtral<br>8x22B|Llama-3-70B-Instruct|GPT3.5-Turbo<br>version 1106|Gemini<br>Pro|GPT-4-Turbo<br>version 1106 (Chat)|
|---------|-----------------------|--------|-------------|-------------------|-------------------|----------|------------------------|
|AGI Eval<br>5-shot|49.7|50.1|54.0|56.9|48.4|49.0|59.6|
|MMLU<br>5-shot|76.6|73.8|76.2|80.2|71.4|66.7|84.0|
|BigBench Hard<br>3-shot|77.9|74.1|81.8|80.4|68.3|75.6|87.7|
|ANLI<br>7-shot|57.3|63.4|65.2|68.3|58.1|64.2|71.7|
|HellaSwag<br>5-shot|81.6|78.0|79.0|82.6|78.8|76.2|88.3|
|ARC Challenge<br>10-shot|91.0|86.9|91.3|93.0|87.4|88.3|95.6|
|ARC Easy<br>10-shot|97.6|95.7|96.9|98.2|96.3|96.1|98.8|
|BoolQ<br>2-shot|86.5|86.1|82.7|89.1|79.1|86.4|91.3|
|CommonsenseQA<br>10-shot|82.2|82.0|82.0|84.4|79.6|81.8|86.7|
|MedQA<br>2-shot|67.6|59.2|67.9|78.5|63.4|58.2|83.7|
|OpenBookQA<br>10-shot|87.2|86.8|88.6|91.8|86.0|86.4|93.4|
|PIQA<br>5-shot|87.8|86.4|85.0|85.3|86.6|86.2|90.1|
|Social IQA<br>5-shot|79.0|75.3|78.2|81.1|68.3|75.4|81.7|
|TruthfulQA (MC2)<br>10-shot|74.3|57.8|67.4|81.9|67.7|72.6|85.2|
|WinoGrande<br>5-shot|78.9|77.0|75.3|83.3|68.8|72.2|86.7|
|TriviaQA<br>5-shot|73.9|82.8|84.5|78.5|85.8|80.2|73.3|
|GSM8K Chain of Thought<br>8-shot|87.5|78.3|83.8|93.5|78.1|80.4|94.2|
|HumanEval<br>0-shot|58.5|61.6|39.6|78.7|62.2|64.4|79.9|
|MBPP<br>3-shot|73.8|68.9|70.7|81.3|77.8|73.2|86.7|
|Average|77.3|75.0|76.3|82.5|74.3|75.4|85.2|

We take a closer look at different categories across 80 public benchmark datasets at the table below:

|Benchmark|Phi-3-Medium-128k-Instruct<br>14b|Command R+<br>104B|Mixtral<br>8x22B|Llama-3-70B-Instruct|GPT3.5-Turbo<br>version 1106|Gemini<br>Pro|GPT-4-Turbo<br>version 1106 (Chat)|
|--------|------------------------|--------|-------------|-------------------|-------------------|----------|------------------------|
| Popular aggregated benchmark | 72.3 | 69.9 | 73.4 | 76.3 | 67.0 | 67.5 | 80.5 |
| Reasoning                    | 83.2 | 79.3 | 81.5 | 86.7 | 78.3 | 80.4 | 89.3 |
| Language understanding       | 75.3 | 75.7 | 78.7 | 77.9 | 70.4 | 75.3 | 81.6 |
| Code generation             | 64.2 | 68.6 | 60.0 | 69.3 | 70.4 | 66.7 | 76.1 |
| Math                         | 52.9 | 45.3 | 52.5 | 59.7 | 52.8 | 50.9 | 67.1 |
| Factual knowledge           | 47.5 | 60.3 | 60.6 | 52.4 | 63.4 | 54.6 | 45.9 |
| Multilingual                | 62.2 | 67.8 | 69.8 | 62.0 | 67.0 | 73.4 | 78.2 |
| Robustness                  | 70.2 | 57.9 | 65.5 | 78.7 | 69.3 | 69.7 | 84.6 |