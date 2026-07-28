We report the results for Phi-3-Small-128K-Instruct on standard open-source benchmarks measuring the model's reasoning ability (both common sense reasoning and logical reasoning). We compare to Mixtral-8x7b, Gemini-Pro, Gemma 7B, Llama-3-8B-Instruct, GPT-3.5-Turbo-1106, and GPT-4-Turbo-1106.

All the reported numbers are produced with the exact same pipeline to ensure that the numbers are comparable. These numbers might differ from other published numbers due to slightly different choices in the evaluation.

As is now standard, we use few-shot prompts to evaluate the models, at temperature 0. 
The prompts and number of shots are part of a Microsoft internal tool to evaluate language models, and in particular we did no optimization to the pipeline for Phi-3.
More specifically, we do not change prompts, pick different few-shot examples, change prompt format, or do any other form of optimization for the model.

The number of k–shot examples is listed per-benchmark. 

|Benchmark|Phi-3-Small-128K-Instruct<br>7b|Gemma<br>7B|Mixtral<br>8x7B|Llama-3-Instruct<br>8b|GPT-3.5-Turbo<br>version 1106|Gemini<br>Pro|GPT-4-Turbo<br>version 1106 (Chat)|
|---------|-------------------------------|----------|-------------|-------------------------|---------------------------|------------|--------------------------------|
|AGI Eval<br>5-shot|43.9|42.1|45.2|42.0|48.4|49.0|59.6|
|MMLU<br>5-shot|75.5|63.6|70.5|66.5|71.4|66.7|84.0|
|BigBench Hard<br>3-shot|77.6|59.6|69.7|51.5|68.3|75.6|87.7|
|ANLI<br>7-shot|55.8|48.7|55.2|57.3|58.1|64.2|71.7|
|HellaSwag<br>5-shot|79.6|49.8|70.4|71.1|78.8|76.2|88.3|
|ARC Challenge<br>10-shot|90.8|78.3|87.3|82.8|87.4|88.3|95.6|
|ARC Easy<br>10-shot|97.3|91.4|95.6|93.4|96.3|96.1|98.8|
|BoolQ<br>2-shot|83.7|66.0|76.6|80.9|79.1|86.4|91.3|
|CommonsenseQA<br>10-shot|80.8|76.2|78.1|79.0|79.6|81.8|86.7|
|MedQA<br>2-shot|46.3|49.6|62.2|60.5|63.4|58.2|83.7|
|OpenBookQA<br>10-shot|87.8|78.6|85.8|82.6|86.0|86.4|93.4|
|PIQA<br>5-shot|88.1|78.1|86.0|75.7|86.6|86.2|90.1|
|Social IQA<br>5-shot|78.7|65.5|75.9|73.9|68.3|75.4|81.7|
|TruthfulQA (MC2)<br>10-shot|69.6|52.1|60.1|63.2|67.7|72.6|85.2|
|WinoGrande<br>5-shot|80.1|55.6|62.0|65.0|68.8|72.2|86.7|
|TriviaQA<br>5-shot|66.0|72.3|82.2|67.7|85.8|80.2|73.3|
|GSM8K Chain of Thought<br>8-shot|87.3|59.8|64.7|77.4|78.1|80.4|94.2|
|HumanEval<br>0-shot|59.1|34.1|37.8|60.4|62.2|64.4|79.9|
|MBPP<br>3-shot|70.3|51.5|60.2|67.7|77.8|73.2|86.7|
|Average|74.6|61.8|69.8|69.4|74.3|75.4|85.2|

We take a closer look at different categories across 80 public benchmark datasets at the table below:

|Benchmark|Phi-3-Small-128K-Instruct<br>7b|Gemma<br>7B|Mixtral<br>8x7B|Llama-3-Instruct<br>8b|GPT-3.5-Turbo<br>version 1106|Gemini<br>Pro|GPT-4-Turbo<br>version 1106 (Chat)|
|--------|--------------------------|--------|-------------|-------------------|-------------------|----------|------------------------|
|Popular aggregated benchmark|70.6|59.4|66.2|59.9|67.0|67.5|80.5|
|Reasoning|80.3|69.1|77.0|75.7|78.3|80.4|89.3|
|Language understanding|67.4|58.4|64.9|65.4|70.4|75.3|81.6|
|Code generation|60.0|45.6|52.7|56.4|70.4|66.7|76.1|
|Math|48.1|35.8|40.3|41.1|52.8|50.9|67.1|
|Factual knowledge|41.7|46.7|58.6|43.1|63.4|54.6|45.9|
|Multilingual|62.6|63.2|63.4|65.0|69.1|76.5|82.0|
|Robustness|68.7|38.4|51.0|64.5|69.3|69.7|84.6|