We evaluated `phi-4` using [OpenAI’s SimpleEval](https://github.com/openai/simple-evals) and our own internal benchmarks to understand the model’s capabilities, more specifically: 

* **MMLU:** Popular aggregated dataset for multitask language understanding.

* **MATH:** Challenging competition math problems.

* **GPQA:** Complex, graduate-level science questions.

* **DROP:** Complex comprehension and reasoning.

* **MGSM:** Multi-lingual grade-school math.

* **HumanEval:** Functional code generation.

* **SimpleQA:** Factual responses.

To understand the capabilities, we compare `phi-4` with a set of models over OpenAI’s SimpleEval benchmark. 

At the high-level overview of the model quality on representative benchmarks. For the table below, higher numbers indicate better performance: 

| **Category**                 | **Benchmark** | **phi-4** (14B) | **phi-3** (14B) | **Qwen 2.5** (14B instruct) | **GPT-4o-mini** | **Llama-3.3** (70B instruct) | **Qwen 2.5** (72B instruct) | **GPT-4o** |
|------------------------------|---------------|-----------|-----------------|----------------------|----------------------|--------------------|-------------------|-----------------|
| Popular Aggregated Benchmark | MMLU          | 84.8      | 77.9            | 79.9                 | 81.8                 | 86.3               | 85.3              | **88.1**            |
| Science                      | GPQA          | **56.1**      | 31.2            | 42.9                 | 40.9                 | 49.1               | 49.0              | 50.6            |
| Math                         | MGSM<br>MATH  | 80.6<br>**80.4** | 53.5<br>44.6 | 79.6<br>75.6 | 86.5<br>73.0 | 89.1<br>66.3* | 87.3<br>80.0              | **90.4**<br>74.6            |
| Code Generation              | HumanEval     | 82.6      | 67.8            | 72.1                 | 86.2                 | 78.9*               | 80.4              | **90.6**            |
| Factual Knowledge            | SimpleQA      | 3.0       | 7.6            | 5.4                 | 9.9                  | 20.9               | 10.2              | **39.4**             |
| Reasoning                    | DROP          | 75.5      | 68.3            | 85.5                 | 79.3                 | **90.2**               | 76.7              | 80.9            |

\* These scores are lower than those reported by Meta, perhaps because simple-evals has a strict formatting requirement that Llama models have particular trouble following. We use the simple-evals framework because it is reproducible, but Meta reports 77 for MATH and 88 for HumanEval on Llama-3.3-70B.
