To understand the capabilities, we compare Phi-3.5-MoE with a set of models over a variety of benchmarks using our internal benchmark platform. At the high-level overview of the model quality on representative benchmarks:

| Category | Benchmark | Phi-3.5-MoE-instruct | Mistral-Nemo-12B-instruct-2407 | Llama-3.1-8B-instruct | Gemma-2-9b-It | Gemini-1.5-Flash | GPT-4o-mini-2024-07-18 (Chat) |
|--|--|--|--|--|--|--|--|
| Popular aggregated benchmark | Arena Hard | 37.9 | 39.4 | 25.7 | 42.0 | 55.2 | 75.0 |
| | BigBench Hard CoT (0-shot) | 79.1 | 60.2 | 63.4 | 63.5 | 66.7 | 80.4 |
| | MMLU (5-shot) | 78.9 | 67.2 | 68.1 | 71.3 | 78.7 | 77.2 |
| | MMLU-Pro (0-shot, CoT) | 54.3 | 40.7 | 44.0 | 50.1 | 57.2 | 62.8 |
| Reasoning | ARC Challenge (10-shot) | 91.0 | 84.8 | 83.1 | 89.8 | 92.8 | 93.5 |
| | BoolQ (2-shot) | 84.6 | 82.5 | 82.8 | 85.7 | 85.8 | 88.7 |
| | GPQA (0-shot, CoT) | 36.8 | 28.6 | 26.3 | 29.2 | 37.5 | 41.1 |
| | HellaSwag (5-shot) | 83.8 | 76.7 | 73.5 | 80.9 | 67.5 | 87.1 |
| | OpenBookQA (10-shot) | 89.6 | 84.4 | 84.8 | 89.6 | 89.0 | 90.0 |
| | PIQA (5-shot) | 88.6 | 83.5 | 81.2 | 83.7 | 87.5 | 88.7 |
| | Social IQA (5-shot) | 78.0 | 75.3 | 71.8 | 74.7 | 77.8 | 82.9 |
| | TruthfulQA (MC2) (10-shot) | 77.5 | 68.1 | 69.2 | 76.6 | 76.6 | 78.2 |
| | WinoGrande (5-shot) | 81.3 | 70.4 | 64.7 | 74.0 | 74.7 | 76.9 |
| Multilingual | Multilingual MMLU (5-shot) | 69.9 | 58.9 | 56.2 | 63.8 | 77.2 | 72.9 |
| | MGSM (0-shot CoT) | 58.7 | 63.3 | 56.7 | 75.1 | 75.8 | 81.7 |
| Math | GSM8K (8-shot, CoT) | 88.7 | 84.2 | 82.4 | 84.9 | 82.4 | 91.3 |
| | MATH (0-shot, CoT) | 59.5 | 31.2 | 47.6 | 50.9 | 38.0 | 70.2 |
| Long context | Qasper | 40.0 | 30.7 | 37.2 | 13.9 | 43.5 | 39.8 |
| | SQuALITY | 24.1 | 25.8 | 26.2 | 0.0 | 23.5 | 23.8 |
| Code Generation | HumanEval (0-shot) | 70.7 | 63.4 | 66.5 | 61.0 | 74.4 | 86.6 |
| | MBPP (3-shot) | 80.8 | 68.1 | 69.4 | 69.3 | 77.5 | 84.1 |
| **Average** | | **69.2** | **61.3** | **61.0** | **63.3** | **68.5** | **74.9** |

We take a closer look at different categories across 80 public benchmark datasets at the table below:
| Category | Phi-3.5-MoE-instruct | Mistral-Nemo-12B-instruct-2407 | Llama-3.1-8B-instruct | Gemma-2-9b-It | Gemini-1.5-Flash | GPT-4o-mini-2024-07-18 (Chat) |
|--|--|--|--|--|--|--|
| Popular aggregated benchmark | 62.6 | 51.9 | 50.3 | 56.7 | 64.5 | 73.9 |
| Reasoning | 78.7 | 72.2 | 70.5 | 75.4 | 77.7 | 80.0 |
| Language understanding | 71.8 | 67.0 | 62.9 | 72.8 | 66.6 | 76.8 |
| Robustness | 75.6 | 65.2 | 59.8 | 64.7 | 68.9 | 77.5 |
| Long context | 25.5 | 24.5 | 25.5 | 0.0 | 27.0 | 25.4 |
| Math | 74.1 | 57.7 | 65.0 | 67.9 | 60.2 | 80.8 |
| Code generation | 68.3 | 56.9 | 65.8 | 58.3 | 66.8 | 69.9 |
| Multilingual | 65.8 | 55.3 | 47.5 | 59.6 | 64.3 | 76.6 |

Overall, Phi-3.5-MoE with only **6.6B active parameters** achieves a similar level of language understanding and math as much larger models. Moreover, the model outperforms bigger models in reasoning capability and only behind GPT-4o-mini. However, it is still fundamentally limited by its size for certain tasks. The model simply does not have the capacity to store too much factual knowledge, therefore, users may experience factual incorrectness. However, we believe such weakness can be resolved by augmenting Phi-3.5 with a search engine, particularly when using the model under RAG settings.

### Multilingual

The table below highlights multilingual capability of Phi-3.5-MoE on multilingual MMLU, MEGA, and multilingual MMLU-pro datasets. Overall, we observed that even with just 6.6B active parameters, the model is very competitive on multilingual tasks in comparison to other models with a much bigger active parameters.

| Category | Phi-3.5-MoE-instruct | Mistral-Nemo-12B-instruct-2407 | Llama-3.1-8B-instruct | Gemma-2-9b-It | Gemini-1.5-Flash | GPT-4o-mini-2024-07-18 (Chat) |
|--|--|--|--|--|--|--|
| Multilingual MMLU | 69.9 | 58.9 | 56.2 | 63.8 | 77.2 | 72.9 |
| Multilingual MMLU-Pro | 45.3 | 34.0 | 21.4 | 43.0 | 57.9 | 53.2 |
| MGSM | 58.7 | 63.3 | 56.7 | 75.1 | 75.8 | 81.7 |
| MEGA MLQA | 65.3 | 61.2 | 45.2 | 54.4 | 61.6 | 70.0 |
| MEGA TyDi QA | 67.1 | 63.7 | 54.5 | 65.6 | 63.6 | 81.8 |
| MEGA UDPOS | 60.4 | 58.2 | 54.1 | 56.6 | 62.4 | 66.0 |
| MEGA XCOPA | 76.6 | 10.8 | 21.1 | 31.2 | 95.0 | 90.3 |
| MEGA XStoryCloze | 82.8 | 92.3 | 71.0 | 87.0 | 20.7 | 96.6 |
| **Average** | **65.8** | **55.3** | **47.5** | **59.6** | **64.3** | **76.6** |

### Long Context

Phi-3.5-MoE supports 128K context length, therefore the model is capable of several long context tasks including long document/meeting summarization, long document QA, multilingual context retrieval. We see that Phi-3.5 is clearly better than Gemma-2 family which only supports 8K context length. Phi-3.5-MoE-instruct is very competitive with other much larger open-weight models such as Llama-3.1-8B-instruct, and Mistral-Nemo-12B-instruct-2407.

| Benchmark | Phi-3.5-MoE-instruct | Mistral-Nemo-12B-instruct-2407 | Llama-3.1-8B-instruct | Gemini-1.5-Flash | GPT-4o-mini-2024-07-18 (Chat) |
|--|--|--|--|--|--|
| GovReport | 26.4 | 25.6 | 25.1 | 27.8 | 24.8 |
| QMSum | 19.9 | 22.1 | 21.6 | 24.0 | 21.7 |
| Qasper | 40.0 | 30.7 | 37.2 | 43.5 | 39.8 |
| SQuALITY | 24.1 | 25.8 | 26.2 | 23.5 | 23.8 |
| SummScreenFD | 16.9 | 18.2 | 17.6 | 16.3 | 17.0 |
| **Average** | **25.5** | **24.5** | **25.5** | **27.0** | **25.4** |

RULER: a retrieval-based benchmark for long context understanding
| Model | 4K | 8K | 16K | 32K | 64K | 128K | Average |
|--|--|--|--|--|--|--|--|
| Phi-3.5-MoE-instruct | 94.8 | 93 | 93.2 | 91.6 | 85.7 | 64.2 | **87.1** |
| Llama-3.1-8B-instruct | 95.5 | 93.8 | 91.6 | 87.4 | 84.7 | 77.0 | **88.3** |
| Mistral-Nemo-12B-instruct-2407 | 87.8 | 87.2 | 87.7 | 69.0 | 46.8 | 19.0 | **66.2** |

RepoQA: a benchmark for long context code understanding
| Model | Python | C++ | Rust | Java | TypeScript | Average |
|--|--|--|--|--|--|--|
| Phi-3.5-MoE-instruct | 89 | 74 | 81 | 88 | 95 | **85** |
| Llama-3.1-8B-instruct | 80 | 65 | 73 | 76 | 63 | **71** |
| Mistral-7B-instruct-v0.3 | 61 | 57 | 51 | 61 | 80 | **62** |
